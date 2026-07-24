@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> scales_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> zps_out: array<u32>;

struct Params {
  num_rows: u32,
  reduce_size: u32,
  quant_min: i32,
  quant_max: i32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 256u;

// Cooperative per-row min/max (mirrors Vulkan choose_qparams_per_row.glsl). One
// workgroup handles a block of up to 4 rows so the 4 int8 zero-points pack into a
// single u32 word with no cross-workgroup write race (zp is int8, elem_size 1).
// Fixed upper bound >= any clamped wg_size; only [0, wg_size) is used.
var<workgroup> part_min: array<f32, 256>;
var<workgroup> part_max: array<f32, 256>;

const SMALL_SCALE_THRESHOLD: f32 = 6.1e-5;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // One workgroup per block of 4 rows; 2D-fold lifts the 65535 grid cap.
  let block = wid.x + wid.y * num_workgroups.x;
  let row0 = block * 4u;
  if (row0 >= params.num_rows) {
    return;
  }
  let lim = min(4u, params.num_rows - row0);
  let qmin = f32(params.quant_min);
  let qmax = f32(params.quant_max);

  var packed_zp: u32 = 0u;
  for (var r: u32 = 0u; r < lim; r = r + 1u) {
    let row = row0 + r;
    let base = row * params.reduce_size;

    // Seed with the row's first element so idle threads contribute a real value.
    var lmin = input[base];
    var lmax = input[base];
    var i = lid.x;
    while (i < params.reduce_size) {
      let v = input[base + i];
      lmin = min(lmin, v);
      lmax = max(lmax, v);
      i = i + wg_size;
    }
    part_min[lid.x] = lmin;
    part_max[lid.x] = lmax;
    workgroupBarrier();

    if (lid.x == 0u) {
      var mn = part_min[0];
      var mx = part_max[0];
      for (var t = 1u; t < wg_size; t = t + 1u) {
        mn = min(mn, part_min[t]);
        mx = max(mx, part_max[t]);
      }

      // calculate_scale_and_zero_point (mirrors Vulkan): extend [min,max] to
      // contain 0, asymmetric scale, error-selected + nudged integer zp.
      mn = min(mn, 0.0);
      mx = max(mx, 0.0);
      var scale = (mx - mn) / (qmax - qmin);
      if (scale == 0.0) {
        scale = 0.1;
      }
      if (scale < SMALL_SCALE_THRESHOLD) {
        let org_scale = scale;
        scale = SMALL_SCALE_THRESHOLD;
        if (mn == 0.0) {
          mx = SMALL_SCALE_THRESHOLD * (qmax - qmin);
        } else if (mx == 0.0) {
          mn = -SMALL_SCALE_THRESHOLD * (qmax - qmin);
        } else {
          let amplifier = SMALL_SCALE_THRESHOLD / org_scale;
          mn = mn * amplifier;
          mx = mx * amplifier;
        }
      }

      let zp_from_min = qmin - mn / scale;
      let zp_from_max = qmax - mx / scale;
      let zp_from_min_error = abs(qmin) - abs(mn / scale);
      let zp_from_max_error = abs(qmax) - abs(mx / scale);
      var initial_zp = zp_from_max;
      if (zp_from_min_error < zp_from_max_error) {
        initial_zp = zp_from_min;
      }
      var nudged: i32;
      if (initial_zp < qmin) {
        nudged = params.quant_min;
      } else if (initial_zp > qmax) {
        nudged = params.quant_max;
      } else {
        nudged = i32(round(initial_zp));
      }

      scales_out[row] = scale;
      packed_zp = packed_zp | ((u32(nudged) & 0xFFu) << (r * 8u));
    }
    // Ensure part_min/part_max are fully consumed before the next row reuses them.
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    zps_out[block] = packed_zp;
  }
}
