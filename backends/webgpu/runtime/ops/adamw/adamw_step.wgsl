
@group(0) @binding(0) var<storage, read_write> t_param: array<f32>;
@group(0) @binding(1) var<storage, read_write> t_m: array<f32>;
@group(0) @binding(2) var<storage, read_write> t_v: array<f32>;
@group(0) @binding(3) var<storage, read> t_grad: array<f32>;

struct Params {
  numel: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  lr: f32,
  beta1: f32,
  beta2: f32,
  eps: f32,
  weight_decay: f32,
  bias_correction1: f32,
  bias_correction2: f32,
  _pad3: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.numel) {
    return;
  }
  let g = t_grad[i];
  var p = t_param[i];
  p = p - params.lr * params.weight_decay * p;
  let m = params.beta1 * t_m[i] + (1.0 - params.beta1) * g;
  let v = params.beta2 * t_v[i] + (1.0 - params.beta2) * g * g;
  t_m[i] = m;
  t_v[i] = v;
  let mhat = m / params.bias_correction1;
  let vhat = v / params.bias_correction2;
  t_param[i] = p - params.lr * mhat / (sqrt(vhat) + params.eps);
}
