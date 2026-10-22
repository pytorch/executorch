struct Params {
  N: u32,
  C: u32,
  IH: u32,
  IW: u32,
  OH: u32,
  OW: u32,
  kH: u32,
  kW: u32,
  sH: u32,
  sW: u32,
  pH: u32,
  pW: u32,
  dH: u32,
  dW: u32,
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<storage, read_write> out_vals: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> out_idx: array<i32>;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u; // = count_x * wg_size; set by host for 2D-spill
override write_indices: u32 = 0u; // 1 = also write out_idx (compile-time gate, mirrors Vulkan)

// max_pool2d (values [+ optional indices]), NCHW row-major, fp32. One thread per
// output element (n, c, oh, ow); gather the window, take the max. General
// stride/pad/dilation. Argmax is ALWAYS tracked (mirrors Vulkan Pool.cpp, which
// computes indices unconditionally and only gates the final write) so the
// values-only and with-indices paths share one kernel; write_indices==0 skips
// only the out_idx store, not the tracking, keeping both paths bit-identical on
// out_vals. Indices are the flat (ih*IW+iw) spatial-plane offset (matches
// torch.nn.functional.max_pool2d(return_indices=True)'s documented convention,
// NOT an absolute offset into the full NCHW tensor). Pad cells are skipped
// (init -inf); out_idx is left at its bound (dummy, when unused) value.
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.N * params.C * params.OH * params.OW;
  let i = gid.y * stride_x + gid.x;
  if (i >= total) {
    return;
  }
  let ow = i % params.OW;
  let oh = (i / params.OW) % params.OH;
  let c = (i / (params.OW * params.OH)) % params.C;
  let n = i / (params.OW * params.OH * params.C);

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  let in_c_base = (n * params.C + c) * params.IH; // * IW added per-row below

  var best: f32 = -3.4e38; // large finite negative max-init (Dawn/Tint rejects -3.40282347e38: > f32 max); matches q4gsw_dq8ca convention
  var best_idx: i32 = 0;
  for (var kh: u32 = 0u; kh < params.kH; kh = kh + 1u) {
    let ih = i32(oh) * i32(params.sH) - i32(params.pH) + i32(kh) * i32(params.dH);
    if (ih < 0 || ih >= iH) {
      continue;
    }
    let in_row = (in_c_base + u32(ih)) * params.IW;
    for (var kw: u32 = 0u; kw < params.kW; kw = kw + 1u) {
      let iw = i32(ow) * i32(params.sW) - i32(params.pW) + i32(kw) * i32(params.dW);
      if (iw < 0 || iw >= iW) {
        continue;
      }
      let v = inp[in_row + u32(iw)];
      if (v > best) {
        best = v;
        best_idx = ih * iW + iw;
      }
    }
  }
  out_vals[i] = best;
  if (write_indices != 0u) {
    out_idx[i] = best_idx;
  }
}
