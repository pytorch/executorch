@group(0) @binding(0) var<storage, read_write> t_out: array<${wgsl_buffer_type(DTYPE, VEC)}>;
@group(0) @binding(1) var<storage, read> t_in: array<${wgsl_buffer_type(DTYPE, VEC)}>;
@group(0) @binding(2) var<storage, read> t_weight: array<${wgsl_buffer_type(DTYPE, VEC)}>;

struct Params {
  num_rows: u32,
  row_width: u32,
  epsilon: f32,
  _pad: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

const WG_SIZE: u32 = 64u;

var<workgroup> shared_sum: array<${wgsl_accum_type()}, WG_SIZE>;

fn reduce_shared(worker_id: u32) {
  workgroupBarrier();
  var stride: u32 = WG_SIZE / 2u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (worker_id < stride) {
      shared_sum[worker_id] = shared_sum[worker_id] + shared_sum[worker_id + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
}

$if VEC == 4:
  // vec4 variant of rms_norm: each lane strides by WG_SIZE over rw4 = row_width/4
  // texels and accumulates dot(v, v). row_width is the ELEMENT count, so mean_sq
  // divides by it (not rw4). The host selects this only when row_width % 4 == 0.
@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>) {
  let row_idx = wid.x;
  let worker_id = lid.x;

  if (row_idx >= params.num_rows) {
    return;
  }

  $if VEC == 4:
    let rw4 = params.row_width / 4u;
    let base4 = row_idx * rw4;
  $else:
    let base = row_idx * params.row_width;

  var local_sq_sum: ${wgsl_accum_type()} = 0.0;
  $if VEC == 4:
    var x4: u32 = worker_id;
    loop {
      if (x4 >= rw4) {
        break;
      }
      let v = t_in[base4 + x4];
      local_sq_sum = local_sq_sum + dot(v, v);
      x4 = x4 + WG_SIZE;
    }
  $else:
    var x: u32 = worker_id;
    loop {
      if (x >= params.row_width) {
        break;
      }
      let v = t_in[base + x];
      local_sq_sum = local_sq_sum + v * v;
      x = x + WG_SIZE;
    }

  shared_sum[worker_id] = local_sq_sum;
  reduce_shared(worker_id);

  let mean_sq = shared_sum[0] / f32(params.row_width);
  let rstd = inverseSqrt(mean_sq + params.epsilon);

  $if VEC == 4:
    x4 = worker_id;
    loop {
      if (x4 >= rw4) {
        break;
      }
      t_out[base4 + x4] = t_in[base4 + x4] * rstd * t_weight[x4];
      x4 = x4 + WG_SIZE;
    }
  $else:
    x = worker_id;
    loop {
      if (x >= params.row_width) {
        break;
      }
      let v = t_in[base + x];
      let w = t_weight[x];
      t_out[base + x] = v * rstd * w;
      x = x + WG_SIZE;
    }
}
