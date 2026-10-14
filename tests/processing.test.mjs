import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const modulePath = fileURLToPath(new URL('../processing.mjs', import.meta.url));

async function loadProcessing() {
  assert.ok(existsSync(modulePath), 'processing.mjs must expose the local image-processing helpers');
  return import(new URL('../processing.mjs', import.meta.url));
}

test('caches model and runtime assets without enabling local model or file-system storage', async () => {
  const { configureRuntimeEnvironment } = await loadProcessing();
  assert.equal(typeof configureRuntimeEnvironment, 'function', 'processing must configure model/runtime caching');
  if (typeof configureRuntimeEnvironment !== 'function') return;

  const env = {
    allowLocalModels: true,
    useFS: true,
    useFSCache: true,
    useBrowserCache: false,
    useWasmCache: false,
    logLevel: 0,
  };
  configureRuntimeEnvironment(env);

  assert.deepEqual(env, {
    allowLocalModels: false,
    useFS: false,
    useFSCache: false,
    useBrowserCache: true,
    useWasmCache: true,
    logLevel: 40,
  });
});

test('accepts JPEG, PNG, and WebP uploads within the documented size limit', async () => {
  const { validateImageFile, MAX_FILE_BYTES } = await loadProcessing();
  for (const type of ['image/jpeg', 'image/png', 'image/webp']) {
    assert.deepEqual(validateImageFile({ type, size: 1024 }), { ok: true });
  }
  assert.deepEqual(validateImageFile(null), { ok: false, reason: 'Choose an image file first.' });
  assert.deepEqual(validateImageFile({ type: 'image/gif', size: 1024 }), {
    ok: false,
    reason: 'Use a JPEG, PNG, or WebP image.',
  });
  assert.deepEqual(validateImageFile({ type: 'image/png', size: MAX_FILE_BYTES + 1 }), {
    ok: false,
    reason: 'This image is larger than 20 MB.',
  });
});

test('converts model logits to finite confidence values between zero and one', async () => {
  const { logitsToConfidence } = await loadProcessing();
  assert.deepEqual(Array.from(logitsToConfidence(new Float32Array([-100, 0, 100]))), [0, 0.5, 1]);
});

test('turns model confidence into a crisp or softly feathered alpha value', async () => {
  const { confidenceToAlpha } = await loadProcessing();
  assert.equal(confidenceToAlpha(0.2, 0), 0);
  assert.equal(confidenceToAlpha(0.8, 0), 1);
  assert.equal(confidenceToAlpha(0.5, 0), 1);
  assert.ok(Math.abs(confidenceToAlpha(0.5, 0.2) - 0.5) < 1e-12);
  assert.equal(confidenceToAlpha(-1, 0.2), 0);
  assert.equal(confidenceToAlpha(2, 0.2), 1);
});

test('resizes a confidence mask with bilinear interpolation', async () => {
  const { resizeMaskBilinear } = await loadProcessing();
  const resized = resizeMaskBilinear(new Float32Array([0, 1, 2, 3]), 2, 2, 3, 3);
  assert.deepEqual(Array.from(resized), [0, 0.5, 1, 1, 1.5, 2, 2, 2.5, 3]);
});

test('applies the mask to alpha while preserving source color and source transparency', async () => {
  const { applyMaskToRgba } = await loadProcessing();
  const pixels = new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 128]);
  const result = applyMaskToRgba(pixels, new Float32Array([0, 1]), 0);
  assert.deepEqual(Array.from(result), [10, 20, 30, 0, 40, 50, 60, 128]);
});

test('selects BiRefNet mask tensors from its raw ONNX output and rejects missing output', async () => {
  const { selectMaskTensor } = await loadProcessing();
  assert.equal(typeof selectMaskTensor, 'function', 'processing must provide a mask-tensor selector');
  if (typeof selectMaskTensor !== 'function') return;

  const tensor = { dims: [1, 1, 512, 512], data: new Float32Array(512 * 512) };
  assert.equal(selectMaskTensor({ output_image: tensor }), tensor);
  assert.equal(selectMaskTensor({ logits: tensor }), tensor);
  assert.throws(() => selectMaskTensor({}), /model output did not include a mask tensor/i);
});

test('locks an image-processing run to its original file until the run finishes', async () => {
  const { beginImageRun, endImageRun } = await loadProcessing();
  assert.equal(typeof beginImageRun, 'function', 'processing must provide an exclusive image-run lock');
  assert.equal(typeof endImageRun, 'function', 'processing must release the image-run lock');
  if (typeof beginImageRun !== 'function' || typeof endImageRun !== 'function') return;

  const firstFile = { name: 'first.png' };
  const replacementFile = { name: 'replacement.png' };
  const state = { file: firstFile, working: false };
  const firstRun = beginImageRun(state);
  assert.strictEqual(firstRun.file, firstFile);
  assert.equal(state.working, true);

  state.file = replacementFile;
  assert.strictEqual(firstRun.file, firstFile);
  assert.equal(beginImageRun(state), null);

  endImageRun(state);
  assert.equal(state.working, false);
  assert.strictEqual(beginImageRun(state).file, replacementFile);
});
