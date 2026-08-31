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

// Direct 2D convolution (non-transposed), NCHW row-major, fp32. ONE thread per
// (b, oc, oh, ow) output element. Supports general stride/padding/dilation and
// groups. input [B,IC,IH,IW]; weight [OC, IC/groups, KH, KW]; bias [OC] (gated).
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

  var acc: f32 = 0.0;
  if (params.has_bias != 0u) {
    acc = bias[oc];
  }

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  for (var icg: u32 = 0u; icg < icpg; icg = icg + 1u) {
    let ic = ic0 + icg;
    let in_c_base = (b * params.IC + ic) * params.IH; // *IW added per-row below
    let w_c_base = (oc * icpg + icg) * params.KH; // *KW added per-row below
    for (var kh: u32 = 0u; kh < params.KH; kh = kh + 1u) {
      let ih = i32(oh) * i32(params.sH) - i32(params.pH) + i32(kh) * i32(params.dH);
      if (ih < 0 || ih >= iH) {
        continue;
      }
      let in_row = (in_c_base + u32(ih)) * params.IW;
      let w_row = (w_c_base + kh) * params.KW;
      for (var kw: u32 = 0u; kw < params.KW; kw = kw + 1u) {
        let iw = i32(ow) * i32(params.sW) - i32(params.pW) + i32(kw) * i32(params.dW);
        if (iw < 0 || iw >= iW) {
          continue;
        }
        acc = acc + input[in_row + u32(iw)] * weight[w_row + kw];
      }
    }
  }
  out[i] = acc;
}
