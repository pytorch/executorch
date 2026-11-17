import { LICENSE_STATUS } from '../config/models.js';

export const WEIGHT_SOURCE = Object.freeze({ REMOTE: 'remote', UNAVAILABLE: 'unavailable' });

// Remote weights require BOTH a cleared license AND an https URL. Either alone
// is treated as "not available" so a stray URL can never imply legal review.
export function resolveWeights(model) {
  const reasons = [];
  if (model?.license?.status !== LICENSE_STATUS.CLEARED) {
    reasons.push({ code: 'license-not-cleared', text: `License review status: ${model?.license?.status ?? 'missing'}` });
  }
  const baseUrl = model?.assets?.baseUrl;
  if (typeof baseUrl !== 'string' || !/^https:\/\//.test(baseUrl)) {
    reasons.push({ code: 'no-asset-url', text: 'No approved download location configured' });
  }
  if (reasons.length) return { source: WEIGHT_SOURCE.UNAVAILABLE, reasons };
  const root = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  return {
    source: WEIGHT_SOURCE.REMOTE,
    reasons: [],
    files: model.assets.files.map((name) => ({ name, url: new URL(name, root).href })),
  };
}
