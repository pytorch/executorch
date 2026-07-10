struct Params {
  M: u32,
  N: u32,
  K: u32,
  has_bias: u32,
};

// input/weight fp32 buffers viewed as vec4 over K (requires K%4==0). Output is
// scalar f32 (+bias). out[m,n] = sum_k input[m,k]*weight[n,k] (+bias[n])
// = sum_k4 dot(in4, w4) (+bias[n]). Same skeleton as the sibling `linear` op's
// linear_vec4.wgsl, with a bias epilogue added.
@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> t_weight: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> t_bias: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const TILE: u32 = 32u;
const TILE4: u32 = 8u;
const RPT: u32 = 4u;

var<workgroup> a_sub: array<array<vec4<f32>, 8>, 32>;
var<workgroup> b_sub: array<array<vec4<f32>, 8>, 32>;

fn in4(row: u32, k4: u32) -> vec4<f32> {
  if (row < params.M && k4 * 4u < params.K) {
    return t_in[row * (params.K / 4u) + k4];
  }
  return vec4<f32>(0.0);
}

fn w4(row: u32, k4: u32) -> vec4<f32> {
  if (row < params.N && k4 * 4u < params.K) {
    return t_weight[row * (params.K / 4u) + k4];
  }
  return vec4<f32>(0.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row0 = wg_id.y * TILE;
  let col0 = wg_id.x * TILE;
  let tr = local_id.y * RPT;
  let tc = local_id.x * RPT;

  var acc: array<array<f32, 4>, 4>;
  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
      acc[ir][ic] = 0.0;
    }
  }

  let num_tiles = (params.K + TILE - 1u) / TILE;
  let flat0 = local_id.y * 8u + local_id.x;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k4s = t * TILE4;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
      let e = flat0 + i * 64u;
      let rr = e / TILE4;
      let kk = e % TILE4;
      a_sub[rr][kk] = in4(row0 + rr, k4s + kk);
      b_sub[rr][kk] = w4(col0 + rr, k4s + kk);
    }
    workgroupBarrier();

    for (var k4: u32 = 0u; k4 < TILE4; k4 = k4 + 1u) {
      for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
        let av = a_sub[tr + ir][k4];
        for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
          acc[ir][ic] = acc[ir][ic] + dot(av, b_sub[tc + ic][k4]);
        }
      }
    }
    workgroupBarrier();
  }

  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
      let r = row0 + tr + ir;
      let c = col0 + tc + ic;
      if (r < params.M && c < params.N) {
        var v = acc[ir][ic];
        if (params.has_bias != 0u) {
          v = v + t_bias[c];
        }
        t_out[r * params.N + c] = v;
      }
    }
  }
}
