@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

struct Params {
  B: u32,
  IC: u32,
  IH: u32,
  IW: u32,
  OC: u32,
  OH: u32,
  OW: u32,
  KH: u32,
  KW: u32,
  sH: u32,
  sW: u32,
  pH: u32,
  pW: u32,
  dH: u32,
  dW: u32,
  groups: u32,
  has_bias: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u; // = count_x * wg_size; set by host for 2D-spill

// Transposed 2D convolution (gather form), NCHW row-major, fp32. ONE thread per
// (b, oc, oh, ow) output element. weight layout = torch convT [IC, OC/groups,
// KH, KW] (NOT flipped). For each kernel tap (kh,kw): an input row ih
// contributes iff (oh + pH - kh*dH) is divisible by sH and ih in range (the
// scatter-inversion). CPU-derisked vs torch.conv_transpose2d to fp64 round-off
// (/tmp/convtr_derisk.py), incl. non-square spatial + non-square kernel.
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.B * params.OC * params.OH * params.OW;
  let i = gid.y * stride_x + gid.x;
  if (i >= total) {
    return;
  }
  let ow = i % params.OW;
  let oh = (i / params.OW) % params.OH;
  let oc = (i / (params.OW * params.OH)) % params.OC;
  let b = i / (params.OW * params.OH * params.OC);

  let icpg = params.IC / params.groups; // input channels per group
  let ocpg = params.OC / params.groups; // output channels per group
  let g = oc / ocpg;
  let ic0 = g * icpg;
  let oc_in_g = oc % ocpg;

  var acc: f32 = 0.0;
  if (params.has_bias != 0u) {
    acc = bias[oc];
  }

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  for (var kh: u32 = 0u; kh < params.KH; kh = kh + 1u) {
    let num_h = i32(oh) + i32(params.pH) - i32(kh) * i32(params.dH);
    if (num_h % i32(params.sH) != 0) {
      continue;
    }
    let ih = num_h / i32(params.sH);
    if (ih < 0 || ih >= iH) {
      continue;
    }
    for (var kw: u32 = 0u; kw < params.KW; kw = kw + 1u) {
      let num_w = i32(ow) + i32(params.pW) - i32(kw) * i32(params.dW);
      if (num_w % i32(params.sW) != 0) {
        continue;
      }
      let iw = num_w / i32(params.sW);
      if (iw < 0 || iw >= iW) {
        continue;
      }
      for (var icg: u32 = 0u; icg < icpg; icg = icg + 1u) {
        let ic = ic0 + icg;
        let in_idx =
            ((b * params.IC + ic) * params.IH + u32(ih)) * params.IW + u32(iw);
        // weight [IC, OC/groups, KH, KW]: index (ic, oc_in_g, kh, kw)
        let w_idx =
            ((ic * ocpg + oc_in_g) * params.KH + kh) * params.KW + kw;
        acc = acc + input[in_idx] * weight[w_idx];
      }
    }
  }
  out[i] = acc;
}
