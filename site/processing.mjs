export const MAX_FILE_BYTES = 20 * 1024 * 1024;
const SUPPORTED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);

export function validateImageFile(file) {
  if (!file) return { ok: false, reason: 'Choose an image file first.' };
  if (!SUPPORTED_TYPES.has(file.type)) {
    return { ok: false, reason: 'Use a JPEG, PNG, or WebP image.' };
  }
  if (file.size > MAX_FILE_BYTES) {
    return { ok: false, reason: 'This image is larger than 20 MB.' };
  }
  return { ok: true };
}

export function logitsToConfidence(logits) {
  const confidence = new Float32Array(logits.length);
  for (let index = 0; index < logits.length; index += 1) {
    const value = logits[index];
    if (value <= -20) confidence[index] = 0;
    else if (value >= 20) confidence[index] = 1;
    else confidence[index] = 1 / (1 + Math.exp(-value));
  }
  return confidence;
}

export function selectMaskTensor(output) {
  const tensor = output?.output_image ?? output?.logits;
  if (!tensor || !Array.isArray(tensor.dims) || !tensor.data || typeof tensor.data.length !== 'number') {
    throw new Error('The model output did not include a mask tensor.');
  }
  return tensor;
}

export function confidenceToAlpha(confidence, feather = 0) {
  const value = Math.min(1, Math.max(0, confidence));
  const softness = Math.min(1, Math.max(0, feather));
  if (softness === 0) return value >= 0.5 ? 1 : 0;

  const start = 0.5 - softness / 2;
  const position = Math.min(1, Math.max(0, (value - start) / softness));
  return position * position * (3 - 2 * position);
}

export function resizeMaskBilinear(mask, sourceWidth, sourceHeight, targetWidth, targetHeight) {
  if (!Number.isInteger(sourceWidth) || !Number.isInteger(sourceHeight) || sourceWidth < 1 || sourceHeight < 1) {
    throw new RangeError('Source mask dimensions must be positive integers.');
  }
  if (!Number.isInteger(targetWidth) || !Number.isInteger(targetHeight) || targetWidth < 1 || targetHeight < 1) {
    throw new RangeError('Target mask dimensions must be positive integers.');
  }
  if (mask.length !== sourceWidth * sourceHeight) {
    throw new RangeError('Mask data length does not match its dimensions.');
  }
  if (sourceWidth === targetWidth && sourceHeight === targetHeight) return Float32Array.from(mask);

  const resized = new Float32Array(targetWidth * targetHeight);
  const scaleX = targetWidth === 1 ? 0 : (sourceWidth - 1) / (targetWidth - 1);
  const scaleY = targetHeight === 1 ? 0 : (sourceHeight - 1) / (targetHeight - 1);

  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = y * scaleY;
    const y0 = Math.floor(sourceY);
    const y1 = Math.min(sourceHeight - 1, y0 + 1);
    const yWeight = sourceY - y0;
    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = x * scaleX;
      const x0 = Math.floor(sourceX);
      const x1 = Math.min(sourceWidth - 1, x0 + 1);
      const xWeight = sourceX - x0;
      const top = mask[y0 * sourceWidth + x0] * (1 - xWeight) + mask[y0 * sourceWidth + x1] * xWeight;
      const bottom = mask[y1 * sourceWidth + x0] * (1 - xWeight) + mask[y1 * sourceWidth + x1] * xWeight;
      resized[y * targetWidth + x] = top * (1 - yWeight) + bottom * yWeight;
    }
  }
  return resized;
}

export function applyMaskToRgba(sourcePixels, confidenceMask, feather = 0) {
  if (sourcePixels.length !== confidenceMask.length * 4) {
    throw new RangeError('RGBA pixels and mask must describe the same image.');
  }
  const result = new Uint8ClampedArray(sourcePixels.length);
  for (let pixel = 0; pixel < confidenceMask.length; pixel += 1) {
    const offset = pixel * 4;
    result[offset] = sourcePixels[offset];
    result[offset + 1] = sourcePixels[offset + 1];
    result[offset + 2] = sourcePixels[offset + 2];
    result[offset + 3] = Math.round(sourcePixels[offset + 3] * confidenceToAlpha(confidenceMask[pixel], feather));
  }
  return result;
}
