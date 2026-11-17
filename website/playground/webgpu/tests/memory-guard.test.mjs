import assert from 'node:assert/strict';
import test from 'node:test';
import { getModel } from '../src/config/models.js';
import { estimatePeakBytes, isMobileDevice, needsMemoryWarning } from '../src/ui/memory-guard.js';

const IPHONE = { userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 Version/26.0 Mobile/15E148 Safari/604.1', maxTouchPoints: 5 };
const IPAD = { userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/26.0 Safari/605.1.15', maxTouchPoints: 5 };
const MAC = { userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/26.0 Safari/605.1.15', maxTouchPoints: 0 };
const ANDROID = { userAgent: 'Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 Chrome/141.0 Mobile Safari/537.36', deviceMemory: 8 };
const LAPTOP_4GB = { userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/141.0 Safari/537.36', deviceMemory: 4 };
const LAPTOP_8GB = { ...LAPTOP_4GB, deviceMemory: 8 };

test('phones and tablets are detected, including iPadOS with a desktop user agent', () => {
  assert.equal(isMobileDevice(IPHONE), true);
  assert.equal(isMobileDevice(IPAD), true);
  assert.equal(isMobileDevice(ANDROID), true);
  assert.equal(isMobileDevice(MAC), false);
  assert.equal(isMobileDevice(LAPTOP_8GB), false);
});

test('chat models count their KV cache in the estimate', () => {
  const qwen = getModel('qwen3-0.6b');
  assert.equal(estimatePeakBytes(qwen), 2 * qwen.approxBytes + 28 * 2 * 8 * 128 * 8960 * 4);
});

test('chat and speech models warn on phones; small vision models do not; desktops never', () => {
  for (const id of ['qwen3-0.6b', 'gemma-3-1b-it', 'whisper-base']) {
    assert.equal(needsMemoryWarning(getModel(id), IPHONE), true, id);
    assert.equal(needsMemoryWarning(getModel(id), MAC), false, id);
    assert.equal(needsMemoryWarning(getModel(id), LAPTOP_8GB), false, id);
  }
  for (const id of ['depth-anything-v2-small', 'modnet', 'real-esrgan', 'dfine-small', 'sam2-hiera-tiny']) {
    assert.equal(needsMemoryWarning(getModel(id), IPHONE), false, id);
  }
  assert.equal(needsMemoryWarning(getModel('qwen3-0.6b'), LAPTOP_4GB), true);
});
