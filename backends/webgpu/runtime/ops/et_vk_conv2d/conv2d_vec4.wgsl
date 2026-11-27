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

// Direct 2D convolution, vec4-gathered over IC (host selects this only when
// icpg % 4 == 0 — NCHW's channel dim has stride IH*IW, not contiguous, so this
// is a register-packing vec4 (4 strided scalar loads combined), not a
// coalesced-memory vec4 load; it still cuts the icg loop trip count 4x. ONE
// thread per (b, oc, oh, ow) output element. input [B,IC,IH,IW];
// weight [OC, IC/groups, KH, KW]; bias [OC] (gated).
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
  let icpg4 = icpg / 4u;

  var acc: f32 = 0.0;
  if (params.has_bias != 0u) {
    acc = bias[oc];
  }

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  let ic_stride = params.IH * params.IW; // channel stride in `input`
  let w_ic_stride = params.KH * params.KW; // channel stride in `weight`
  for (var icg4: u32 = 0u; icg4 < icpg4; icg4 = icg4 + 1u) {
    let icg = icg4 * 4u;
    let ic = ic0 + icg;
    let in_c_base = b * params.IC * ic_stride + ic * ic_stride;
    let w_c_base = oc * icpg * w_ic_stride + icg * w_ic_stride;
    for (var kh: u32 = 0u; kh < params.KH; kh = kh + 1u) {
      let ih = i32(oh) * i32(params.sH) - i32(params.pH) + i32(kh) * i32(params.dH);
      if (ih < 0 || ih >= iH) {
        continue;
      }
      for (var kw: u32 = 0u; kw < params.KW; kw = kw + 1u) {
        let iw = i32(ow) * i32(params.sW) - i32(params.pW) + i32(kw) * i32(params.dW);
        if (iw < 0 || iw >= iW) {
          continue;
        }
        let in_off = u32(ih) * params.IW + u32(iw);
        let w_off = kh * params.KW + kw;
        let in4 = vec4<f32>(
            input[in_c_base + in_off],
            input[in_c_base + ic_stride + in_off],
            input[in_c_base + 2u * ic_stride + in_off],
            input[in_c_base + 3u * ic_stride + in_off]);
        let w4 = vec4<f32>(
            weight[w_c_base + w_off],
            weight[w_c_base + w_ic_stride + w_off],
            weight[w_c_base + 2u * w_ic_stride + w_off],
            weight[w_c_base + 3u * w_ic_stride + w_off]);
        acc = acc + dot(in4, w4);
      }
    }
  }
  out[i] = acc;
}
