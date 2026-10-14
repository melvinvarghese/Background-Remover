import {
  applyMaskToRgba,
  logitsToConfidence,
  resizeMaskBilinear,
  selectMaskTensor,
  validateImageFile,
} from './processing.mjs';

const MODEL_ID = 'studioludens/birefnet-lite-512';
const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2';
const MAX_IMAGE_PIXELS = 25_000_000;

const elements = {
  fileInput: document.querySelector('#imageInput'),
  dropzone: document.querySelector('#dropzone'),
  introDropzone: document.querySelector('#introDropzone'),
  workbench: document.querySelector('#workbench'),
  emptyPrompt: document.querySelector('#emptyPrompt'),
  imageWorkspace: document.querySelector('#imageWorkspace'),
  sourcePreview: document.querySelector('#sourcePreview'),
  sourceLabel: document.querySelector('#sourceLabel'),
  resultCanvas: document.querySelector('#resultCanvas'),
  processButton: document.querySelector('#processButton'),
  downloadButton: document.querySelector('#downloadButton'),
  clearButton: document.querySelector('#clearButton'),
  edgeControl: document.querySelector('#edgeControl'),
  edgeSlider: document.querySelector('#edgeSlider'),
  edgeValue: document.querySelector('#edgeValue'),
  status: document.querySelector('#status'),
  progress: document.querySelector('#progress'),
  progressLabel: document.querySelector('#progressLabel'),
};

const state = {
  file: null,
  objectUrl: null,
  originalPixels: null,
  originalWidth: 0,
  originalHeight: 0,
  confidenceMask: null,
  model: null,
  processor: null,
  RawImage: null,
  working: false,
};

function setStatus(message, kind = 'info') {
  elements.status.textContent = message;
  elements.status.dataset.kind = kind;
}

function setProgress(value, message) {
  elements.progress.hidden = false;
  elements.progress.value = value;
  elements.progressLabel.textContent = message;
}

function hideProgress() {
  elements.progress.hidden = true;
  elements.progress.value = 0;
  elements.progressLabel.textContent = '';
}

function clearResult() {
  state.originalPixels = null;
  state.confidenceMask = null;
  state.originalWidth = 0;
  state.originalHeight = 0;
  elements.resultCanvas.width = 1;
  elements.resultCanvas.height = 1;
  elements.downloadButton.hidden = true;
  elements.edgeControl.hidden = true;
  elements.imageWorkspace.hidden = true;
}

function chooseFile(file) {
  const validation = validateImageFile(file);
  if (!validation.ok) {
    setStatus(validation.reason, 'error');
    return;
  }

  clearResult();
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = file;
  state.objectUrl = URL.createObjectURL(file);
  elements.sourcePreview.src = state.objectUrl;
  elements.sourceLabel.textContent = file.name;
  elements.emptyPrompt.hidden = true;
  elements.imageWorkspace.hidden = false;
  elements.introDropzone.hidden = true;
  elements.dropzone.hidden = true;
  elements.clearButton.hidden = false;
  elements.processButton.disabled = false;
  setStatus('Image ready. It stays in this browser tab.');
  hideProgress();
}

function resetApp() {
  clearResult();
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = null;
  state.objectUrl = null;
  elements.fileInput.value = '';
  elements.sourcePreview.removeAttribute('src');
  elements.sourceLabel.textContent = '';
  elements.emptyPrompt.hidden = false;
  elements.introDropzone.hidden = false;
  elements.dropzone.hidden = false;
  elements.clearButton.hidden = true;
  elements.processButton.disabled = true;
  elements.processButton.textContent = 'Remove background';
  setStatus('Choose a photo to begin.');
  hideProgress();
}

function progressForFile(progress) {
  if (progress?.status === 'progress' && Number.isFinite(progress.loaded) && Number.isFinite(progress.total) && progress.total > 0) {
    const percent = Math.min(100, Math.round((progress.loaded / progress.total) * 100));
    const loaded = Math.round(progress.loaded / (1024 * 1024));
    const total = Math.ceil(progress.total / (1024 * 1024));
    setProgress(percent, `Loading the model · ${loaded} of ${total} MB`);
  } else if (progress?.status === 'initiate') {
    setProgress(0, 'Connecting to the model host…');
  }
}

async function getModelAndProcessor() {
  if (state.model && state.processor && state.RawImage) return;

  setStatus('Loading the image model. Your photo stays in this tab.');
  setProgress(0, 'Loading the browser runtime…');
  const { AutoModel, AutoProcessor, RawImage, env } = await import(TRANSFORMERS_URL);

  // The runtime may fetch app/model files, but the user-selected File is only decoded locally.
  env.allowLocalModels = false;
  env.useFS = false;
  env.useFSCache = false;
  env.useBrowserCache = false;
  env.useWasmCache = false;
  env.logLevel = 40;

  let device = 'wasm';
  try {
    if (navigator.gpu && await navigator.gpu.requestAdapter()) device = 'webgpu';
  } catch {
    device = 'wasm';
  }

  const options = { dtype: 'fp16', device, progress_callback: progressForFile };
  try {
    state.model = await AutoModel.from_pretrained(MODEL_ID, options);
  } catch (error) {
    if (device !== 'webgpu') throw error;
    setStatus('GPU setup did not work on this device. Retrying with the browser CPU runtime.');
    setProgress(0, 'Starting the CPU model…');
    state.model = await AutoModel.from_pretrained(MODEL_ID, {
      ...options,
      device: 'wasm',
    });
  }

  setProgress(100, 'Preparing the image processor…');
  state.processor = await AutoProcessor.from_pretrained(MODEL_ID, { progress_callback: progressForFile });
  state.RawImage = RawImage;
  hideProgress();
}

function renderCutout() {
  if (!state.originalPixels || !state.confidenceMask) return;
  const feather = Number(elements.edgeSlider.value) / 100;
  const rgba = applyMaskToRgba(state.originalPixels, state.confidenceMask, feather);
  const canvas = elements.resultCanvas;
  canvas.width = state.originalWidth;
  canvas.height = state.originalHeight;
  const context = canvas.getContext('2d', { willReadFrequently: false });
  context.putImageData(new ImageData(rgba, state.originalWidth, state.originalHeight), 0, 0);
  elements.edgeValue.textContent = `${elements.edgeSlider.value}%`;
  elements.downloadButton.hidden = false;
  elements.edgeControl.hidden = false;
}

async function processImage() {
  if (!state.file || state.working) return;
  state.working = true;
  elements.processButton.disabled = true;
  elements.processButton.textContent = 'Preparing…';
  elements.downloadButton.hidden = true;
  elements.edgeControl.hidden = true;

  let bitmap;
  try {
    bitmap = await createImageBitmap(state.file);
    const { width, height } = bitmap;
    if (width * height > MAX_IMAGE_PIXELS) {
      throw new Error('This photo is over 25 megapixels. Resize it and try again.');
    }

    const sourceCanvas = document.createElement('canvas');
    sourceCanvas.width = width;
    sourceCanvas.height = height;
    const sourceContext = sourceCanvas.getContext('2d', { willReadFrequently: true });
    if (!sourceContext) throw new Error('This browser could not prepare the photo. Try a current version of Chrome, Edge, Firefox, or Safari.');
    sourceContext.drawImage(bitmap, 0, 0, width, height);
    state.originalPixels = sourceContext.getImageData(0, 0, width, height).data;
    state.originalWidth = width;
    state.originalHeight = height;

    await getModelAndProcessor();
    setStatus('Finding the subject locally…');
    setProgress(0, 'Preparing a 512 × 512 model input…');

    const rawImage = await state.RawImage.read(state.file);
    const { pixel_values: pixelValues } = await state.processor(rawImage);
    const modelOutput = await state.model({ input_image: pixelValues });
    const maskTensor = selectMaskTensor(modelOutput);
    const dims = maskTensor.dims;
    const maskHeight = dims[dims.length - 2];
    const maskWidth = dims[dims.length - 1];
    if (maskTensor.data.length !== maskWidth * maskHeight) {
      throw new Error('The model returned an unexpected mask. Please try another image.');
    }

    const confidence = logitsToConfidence(maskTensor.data);
    state.confidenceMask = resizeMaskBilinear(confidence, maskWidth, maskHeight, width, height);
    renderCutout();
    setStatus(`Cutout ready · ${width.toLocaleString()} × ${height.toLocaleString()} px PNG`);
    setProgress(100, 'Done');
    window.setTimeout(hideProgress, 900);
  } catch (error) {
    state.originalPixels = null;
    state.confidenceMask = null;
    elements.downloadButton.hidden = true;
    elements.edgeControl.hidden = true;
    setStatus(error?.message || 'The image could not be processed. Try a different photo or browser.', 'error');
    hideProgress();
  } finally {
    bitmap?.close();
    state.working = false;
    elements.processButton.disabled = !state.file;
    elements.processButton.textContent = 'Remove background';
  }
}

function downloadPng() {
  if (!state.originalPixels || !state.confidenceMask) return;
  elements.resultCanvas.toBlob((blob) => {
    if (!blob) {
      setStatus('The PNG could not be created. Try again.', 'error');
      return;
    }
    const downloadUrl = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = downloadUrl;
    anchor.download = `${state.file.name.replace(/\.[^.]+$/, '') || 'cutout'}-transparent.png`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);
    setStatus('Transparent PNG downloaded to your device.');
  }, 'image/png');
}

function handleDrop(event) {
  event.preventDefault();
  elements.workbench.classList.remove('is-dragging');
  chooseFile(event.dataTransfer?.files?.[0]);
}

for (const target of [elements.dropzone, elements.workbench]) {
  target.addEventListener('dragover', (event) => {
    event.preventDefault();
    elements.workbench.classList.add('is-dragging');
  });
  target.addEventListener('dragleave', (event) => {
    if (!elements.workbench.contains(event.relatedTarget)) elements.workbench.classList.remove('is-dragging');
  });
  target.addEventListener('drop', handleDrop);
}

elements.fileInput.addEventListener('change', () => {
  chooseFile(elements.fileInput.files?.[0]);
  elements.fileInput.value = '';
});
elements.processButton.addEventListener('click', processImage);
elements.downloadButton.addEventListener('click', downloadPng);
elements.clearButton.addEventListener('click', resetApp);
elements.edgeSlider.addEventListener('input', renderCutout);

setStatus('Choose a photo to begin.');
