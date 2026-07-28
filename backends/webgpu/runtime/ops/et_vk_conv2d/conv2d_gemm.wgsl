@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

struct Params {
  B: u32, IC: u32, IH: u32, IW: u32,
  OC: u32, OH: u32, OW: u32, KH: u32, KW: u32,
  sH: u32, sW: u32, pH: u32, pW: u32, dH: u32, dW: u32,
  groups: u32, has_bias: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

// Standard (groups==1) conv2d as an implicit-im2col tiled GEMM, reusing the
// linear_fp32_tiled skeleton (TILE=32, RPT=4, shared-mem tiles). M=OC, N=B*OH*OW,
// K=IC*KH*KW. A=weight [OC,K] (contiguous OIHW); B=input sampled on the fly (im2col),
// out-of-range -> 0.0 implements padding. bias is per-ROW (OC). Output written NCHW.
const TILE: u32 = 32u;
const RPT: u32 = 4u;

var<workgroup> a_sub: array<array<f32, 32>, 32>;
var<workgroup> b_sub: array<array<f32, 32>, 32>;

fn gK() -> u32 { return params.IC * params.KH * params.KW; }
fn gN() -> u32 { return params.B * params.OH * params.OW; }

fn read_a(m: u32, kk: u32) -> f32 { // weight[OC, IC*KH*KW] row-major
  if (m < params.OC && kk < gK()) {
    return weight[m * gK() + kk];
  }
  return 0.0;
}

fn read_b(kk: u32, n: u32) -> f32 { // im2col sample of input[B,IC,IH,IW]
  if (kk >= gK() || n >= gN()) {
    return 0.0;
  }
  let ohw = params.OH * params.OW;
  let b = n / ohw;
  let sp = n % ohw;
  let oh = sp / params.OW;
  let ow = sp % params.OW;
  let khw = params.KH * params.KW;
  let ic = kk / khw;
  let r = kk % khw;
  let kh = r / params.KW;
  let kw = r % params.KW;
  let ih = i32(oh) * i32(params.sH) - i32(params.pH) + i32(kh) * i32(params.dH);
  let iw = i32(ow) * i32(params.sW) - i32(params.pW) + i32(kw) * i32(params.dW);
  if (ih < 0 || ih >= i32(params.IH) || iw < 0 || iw >= i32(params.IW)) {
    return 0.0; // padding
  }
  return input[((b * params.IC + ic) * params.IH + u32(ih)) * params.IW + u32(iw)];
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
  let M = params.OC;
  let N = gN();
  let K = gK();
  let tile_row0 = wg_id.y * TILE;
  let tile_col0 = wg_id.x * TILE;
  let tile_row = local_id.y * RPT;
  let tile_col = local_id.x * RPT;

  var acc: array<array<f32, 4>, 4>;
  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
      acc[ir][ic] = 0.0;
    }
  }

  let num_tiles = (K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k_start = t * TILE;
    for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
      let arow = local_id.y * RPT + ir;
      for (var kk: u32 = 0u; kk < RPT; kk = kk + 1u) {
        let col = local_id.x * RPT + kk;
        a_sub[arow][col] = read_a(tile_row0 + arow, k_start + col);
        b_sub[arow][col] = read_b(k_start + arow, tile_col0 + col);
      }
    }
    workgroupBarrier();
    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
        let aval = a_sub[tile_row + ir][k];
        for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
          acc[ir][ic] = acc[ir][ic] + aval * b_sub[k][tile_col + ic];
        }
      }
    }
    workgroupBarrier();
  }

  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
      let m = tile_row0 + tile_row + ir; // oc
      let n = tile_col0 + tile_col + ic; // spatial index
      if (m < M && n < N) {
        var val = acc[ir][ic];
        if (params.has_bias != 0u) {
          val = val + bias[m];
        }
        let ohw = params.OH * params.OW;
        let b = n / ohw;
        let sp = n % ohw;
        let oh = sp / params.OW;
        let ow = sp % params.OW;
        out[((b * params.OC + m) * params.OH + oh) * params.OW + ow] = val;
      }
    }
  }
}
