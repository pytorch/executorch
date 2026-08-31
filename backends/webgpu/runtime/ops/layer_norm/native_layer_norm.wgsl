@group(0) @binding(0) var<storage, read_write> t_out: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> t_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> t_weight: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> t_bias: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> t_mean: array<f32>;
@group(0) @binding(5) var<storage, read_write> t_rstd: array<f32>;

struct Params {
  num_rows: u32,
  row_width: u32,
  epsilon: f32,
  has_affine: u32,
}
@group(0) @binding(6) var<uniform> params: Params;

const WG_SIZE: u32 = 64u;

override stride_x: u32 = 4294967295u; // = count_x; set by host for 2D-spill

// Single-pass, numerically-robust mean+variance via Chan et al.'s parallel
// Welford: each thread folds its strided vec4<f32> elements into a running
// (n, mean, M2), then this tree-reduces the WG_SIZE per-thread triples via
// pairwise merges — no naive E[x^2]-E[x]^2 cancellation risk, de-risked on
// CPU against large-mean/small-variance activations.
var<workgroup> shared_n: array<f32, WG_SIZE>;
var<workgroup> shared_mean: array<f32, WG_SIZE>;
var<workgroup> shared_m2: array<f32, WG_SIZE>;

fn reduce_shared_welford(worker_id: u32) {
  workgroupBarrier();
  var stride: u32 = WG_SIZE / 2u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (worker_id < stride) {
      let na = shared_n[worker_id];
      let nb = shared_n[worker_id + stride];
      let n = na + nb;
      if (n > 0.0) {
        let delta = shared_mean[worker_id + stride] - shared_mean[worker_id];
        shared_mean[worker_id] = shared_mean[worker_id] + delta * (nb / n);
        shared_m2[worker_id] = shared_m2[worker_id] + shared_m2[worker_id + stride]
            + delta * delta * (na * nb / n);
        shared_n[worker_id] = n;
      }
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>) {
  let row_idx = wid.y * stride_x + wid.x;
  let worker_id = lid.x;

  if (row_idx >= params.num_rows) {
    return;
  }

  let row_width4 = params.row_width / 4u;
  let base4 = row_idx * row_width4;

  var local_n: f32 = 0.0;
  var local_mean: f32 = 0.0;
  var local_m2: f32 = 0.0;
  var x4: u32 = worker_id;
  loop {
    if (x4 >= row_width4) {
      break;
    }
    let v4 = t_in[base4 + x4];
    for (var c: u32 = 0u; c < 4u; c = c + 1u) {
      local_n = local_n + 1.0;
      let v = v4[c];
      let delta = v - local_mean;
      local_mean = local_mean + delta / local_n;
      let delta2 = v - local_mean;
      local_m2 = local_m2 + delta * delta2;
    }
    x4 = x4 + WG_SIZE;
  }
  shared_n[worker_id] = local_n;
  shared_mean[worker_id] = local_mean;
  shared_m2[worker_id] = local_m2;
  reduce_shared_welford(worker_id);
  let mean = shared_mean[0];
  let variance = shared_m2[0] / f32(params.row_width);
  let rstd = inverseSqrt(variance + params.epsilon);

  if (worker_id == 0u) {
    t_mean[row_idx] = mean;
    t_rstd[row_idx] = rstd;
  }
  workgroupBarrier();

  x4 = worker_id;
  loop {
    if (x4 >= row_width4) {
      break;
    }
    let v4 = t_in[base4 + x4];
    let normed = (v4 - vec4<f32>(mean)) * rstd;
    // weight/bias are optional (None when this is the group_norm LN-reframe); the
    // dummy buffers are not bound then, so gate the affine on has_affine.
    if (params.has_affine == 1u) {
      t_out[base4 + x4] = normed * t_weight[x4] + t_bias[x4];
    } else {
      t_out[base4 + x4] = normed;
    }
    x4 = x4 + WG_SIZE;
  }
}
