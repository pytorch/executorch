@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_input: array<f32>;
@group(0) @binding(2) var<storage, read> t_weight: array<u32>;
@group(0) @binding(3) var<storage, read> t_scales: array<f32>;
@group(0) @binding(4) var<storage, read> t_bias: array<f32>;

struct Params {
  M: u32,
  N: u32,
  K: u32,
  K_packed: u32,
  group_size: u32,
  padded_N: u32,
  has_bias: u32,
  _pad: u32,
}
@group(0) @binding(5) var<uniform> params: Params;

// "steel" prefill GEMM (M>1): 64x64 tile, 256 threads; K%16==0 host-guarded.
// The "steel" name + register-tiled dequant-to-shared GEMM structure are
// inspired by MLX's steel GEMM kernels (github.com/ml-explore/mlx,
// mlx/backend/metal/kernels/steel).
const BM: u32 = 64u; const BN: u32 = 64u; const BK: u32 = 16u;
var<workgroup> As: array<${buffer_scalar_type(DTYPE)}, 1024>;   // BM*BK
var<workgroup> Bs: array<${buffer_scalar_type(DTYPE)}, 1024>;   // BK*BN
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let nbN = (params.N + BN - 1u) / BN;
  let bx = wid.x % nbN;      // decode 2D tile id from 1D dispatch
  let by = wid.x / nbN;
  let row0 = by * BM;
  let col0 = bx * BN;
  let tid = lid.y * 16u + lid.x;
  var acc: array<array<f32, 4>, 4>;
  for (var m: u32 = 0u; m < 4u; m = m + 1u) {
    for (var n: u32 = 0u; n < 4u; n = n + 1u) { acc[m][n] = 0.0; }
  }
  // A staging coords: 256 threads load 64x16 = 1024 f32 -> 4 rows each (4 contiguous K).
  let ar = tid / 4u;          // 0..63 (row in tile)
  let ac = (tid % 4u) * 4u;   // 0,4,8,12 (K offset, 4 contiguous)
  // B staging coords: 256 threads load 16x64 = 1024 dequant weights -> 4 cols each.
  let br = tid / 16u;         // 0..15 (K within BK)
  let bc = (tid % 16u) * 4u;  // 0,4,..60 (N offset, 4 contiguous)

  var k0: u32 = 0u;
  loop {
    if (k0 >= params.K) { break; }
    // stage activations (edge-masked on M; K is a multiple of BK for our shapes)
    let arow = row0 + ar;
    if (arow < params.M) {
      let base = arow * params.K + k0 + ac;
      As[ar * BK + ac + 0u] = ${buffer_scalar_type(DTYPE)}(t_input[base]);
      As[ar * BK + ac + 1u] = ${buffer_scalar_type(DTYPE)}(t_input[base + 1u]);
      As[ar * BK + ac + 2u] = ${buffer_scalar_type(DTYPE)}(t_input[base + 2u]);
      As[ar * BK + ac + 3u] = ${buffer_scalar_type(DTYPE)}(t_input[base + 3u]);
    } else {
      As[ar * BK + ac + 0u] = 0.0; As[ar * BK + ac + 1u] = 0.0;
      As[ar * BK + ac + 2u] = 0.0; As[ar * BK + ac + 3u] = 0.0;
    }
    // stage DEQUANTIZED weights into Bs[k][n]: 4 contiguous N per thread.
    let kk = k0 + br;               // K index for this shmem row
    let scale_row = (kk / params.group_size) * params.padded_N;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      let n = col0 + bc + j;
      var dqv: ${buffer_scalar_type(DTYPE)} = 0.0;
      if (n < params.N) {
        let byte_idx = n * params.K_packed + (kk >> 1u);
        let word = t_weight[byte_idx >> 2u];
        let b = (word >> ((byte_idx & 3u) * 8u)) & 0xFFu;
        var nib: u32;
        if ((kk & 1u) == 0u) { nib = b & 0x0Fu; } else { nib = (b >> 4u) & 0x0Fu; }
        dqv = f32(i32(nib) - 8) * t_scales[scale_row + n];
      }
      Bs[br * BN + bc + j] = dqv;
    }
    workgroupBarrier();
    for (var k: u32 = 0u; k < BK; k = k + 1u) {
      var a: array<${buffer_scalar_type(DTYPE)}, 4>;
      var bvec: array<${buffer_scalar_type(DTYPE)}, 4>;
      for (var m: u32 = 0u; m < 4u; m = m + 1u) { a[m] = As[(lid.y * 4u + m) * BK + k]; }
      for (var n: u32 = 0u; n < 4u; n = n + 1u) { bvec[n] = Bs[k * BN + lid.x * 4u + n]; }
      for (var m: u32 = 0u; m < 4u; m = m + 1u) {
        for (var n: u32 = 0u; n < 4u; n = n + 1u) { acc[m][n] = acc[m][n] + a[m] * bvec[n]; }
      }
    }
    workgroupBarrier();
    k0 = k0 + BK;
  }
  for (var m: u32 = 0u; m < 4u; m = m + 1u) {
    for (var n: u32 = 0u; n < 4u; n = n + 1u) {
      let r = row0 + lid.y * 4u + m;
      let c = col0 + lid.x * 4u + n;
      if (r < params.M && c < params.N) {
        var v = acc[m][n];
        if (params.has_bias != 0u) { v = v + t_bias[c]; }
        t_out[r * params.N + c] = v;
      }
    }
  }
}
