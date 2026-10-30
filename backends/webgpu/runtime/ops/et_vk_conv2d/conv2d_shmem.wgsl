@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

struct Params {
  B: u32, IC: u32, IH: u32, IW: u32,
  OC: u32, OH: u32, OW: u32, _kh: u32, _kw: u32,
  sH: u32, sW: u32, pH: u32, pW: u32, dH: u32, dW: u32,
  groups: u32, has_bias: u32, has_clamp: u32, out_min: f32, out_max: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

// groups==1 conv2d, direct form with the input tile staged in shared memory.
// A workgroup owns TM output channels x a TH x TW output tile of one image.
// Per chunk of ICT input channels it stages the tile's input footprint (HH x HW
// incl. the kernel halo) and the matching weights once, then every KH*KW tap
// reads from shared memory, so each input element is fetched ~once instead of
// KH*KW times as in an im2col GEMM. 64 threads: WG_M groups of 4 channels x
// WG_P groups of PPT consecutive output pixels along W.
override WG_M: u32 = 8u;
override PPT: u32 = 8u;
override TW: u32 = 16u;
override ICT: u32 = 8u;
override KH: u32 = 3u;
override KW: u32 = 3u;
override HH: u32 = 6u; // (TH-1)*sH + (KH-1)*dH + 1
override HW: u32 = 18u; // (TW-1)*sW + (KW-1)*dW + 1
override WG_P: u32 = 64u / WG_M;
override TM: u32 = 4u * WG_M;
override TH: u32 = WG_P * PPT / TW;

var<workgroup> in_s: array<f32, ICT * HH * HW>; // [ic][hy][hx]
var<workgroup> w_s: array<f32, ICT * KH * KW * TM>; // [ic][kh][kw][m]

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32) {
  let tiles_w = (params.OW + TW - 1u) / TW;
  let tiles_h = (params.OH + TH - 1u) / TH;
  let m0 = (wg_id.x / tiles_w) * TM;
  let ow0 = (wg_id.x % tiles_w) * TW;
  let b = wg_id.y / tiles_h;
  let oh0 = (wg_id.y % tiles_h) * TH;
  let ih0 = i32(oh0 * params.sH) - i32(params.pH);
  let iw0 = i32(ow0 * params.sW) - i32(params.pW);

  let ty = lid / WG_P;
  let p = lid % WG_P;
  let r = (p * PPT) / TW;
  let c0 = (p * PPT) % TW;
  let ihw = params.IH * params.IW;

  var acc: array<vec4<f32>, 16>; // [pixel], each vec4 = 4 output channels
  for (var j: u32 = 0u; j < PPT; j = j + 1u) {
    acc[j] = vec4<f32>(0.0);
  }

  for (var icb: u32 = 0u; icb < params.IC; icb = icb + ICT) {
    for (var e: u32 = lid; e < ICT * HH * HW; e = e + 64u) {
      let ic = icb + e / (HH * HW);
      let rem = e % (HH * HW);
      let ih = ih0 + i32(rem / HW);
      let iw = iw0 + i32(rem % HW);
      var v = 0.0;
      if (ic < params.IC && ih >= 0 && ih < i32(params.IH) && iw >= 0 &&
          iw < i32(params.IW)) {
        v = input[(b * params.IC + ic) * ihw + u32(ih) * params.IW + u32(iw)];
      }
      in_s[e] = v;
    }
    for (var e: u32 = lid; e < ICT * KH * KW * TM; e = e + 64u) {
      let ml = e / (ICT * KH * KW);
      let rem = e % (ICT * KH * KW);
      let icl = rem / (KH * KW);
      let tap = rem % (KH * KW);
      let m = m0 + ml;
      let ic = icb + icl;
      var v = 0.0;
      if (m < params.OC && ic < params.IC) {
        v = weight[(m * params.IC + ic) * KH * KW + tap];
      }
      w_s[(icl * KH * KW + tap) * TM + ml] = v;
    }
    workgroupBarrier();

    for (var icl: u32 = 0u; icl < ICT; icl = icl + 1u) {
      for (var kh: u32 = 0u; kh < KH; kh = kh + 1u) {
        let row = icl * HH * HW + (r * params.sH + kh * params.dH) * HW;
        for (var kw: u32 = 0u; kw < KW; kw = kw + 1u) {
          let wi = ((icl * KH + kh) * KW + kw) * TM + ty * 4u;
          let wv = vec4<f32>(w_s[wi], w_s[wi + 1u], w_s[wi + 2u], w_s[wi + 3u]);
          let col = row + c0 * params.sW + kw * params.dW;
          for (var j: u32 = 0u; j < PPT; j = j + 1u) {
            acc[j] = fma(wv, vec4<f32>(in_s[col + j * params.sW]), acc[j]);
          }
        }
      }
    }
    workgroupBarrier();
  }

  let oh = oh0 + r;
  if (b >= params.B || oh >= params.OH) {
    return;
  }
  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    let m = m0 + ty * 4u + i;
    if (m >= params.OC) {
      continue;
    }
    var bias_v = 0.0;
    if (params.has_bias != 0u) {
      bias_v = bias[m];
    }
    let out_row = ((b * params.OC + m) * params.OH + oh) * params.OW;
    for (var j: u32 = 0u; j < PPT; j = j + 1u) {
      let ow = ow0 + c0 + j;
      if (ow < params.OW) {
        var val = acc[j][i] + bias_v;
        if (params.has_clamp != 0u) {
          val = clamp(val, params.out_min, params.out_max);
        }
        out[out_row + ow] = val;
      }
    }
  }
}
