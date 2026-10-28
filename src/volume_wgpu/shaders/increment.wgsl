struct Params {
  value: f32,
};

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let index = global_id.x;
  let count = arrayLength(&input_data);
  if (index >= count) {
    return;
  }
  output_data[index] = input_data[index] + params.value;
}
