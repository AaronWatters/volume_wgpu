@group(0) @binding(0) var<storage, read_write> data_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> depth_a: array<f32>;
@group(0) @binding(2) var<uniform> geometry_a: DepthBufferGeometry;

@group(1) @binding(0) var<storage, read_write> data_b: array<f32>;
@group(1) @binding(1) var<storage, read_write> depth_b: array<f32>;
@group(1) @binding(2) var<uniform> geometry_b: DepthBufferGeometry;

@group(2) @binding(0) var<storage, read_write> output_data: array<f32>;
@group(2) @binding(1) var<storage, read_write> output_depth: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let b_offset = global_id.x;
  if (b_offset >= arrayLength(&depth_b)) {
    return;
  }

  let b_location = L_location(vec3<u32>(offset_to_L(b_offset, geometry_b), 0u), geometry_b);
  let b_start = b_location.X;
  let b_end = b_start + vec3<f32>(geometry_b.dx, geometry_b.dy, 0.0);
  let a_extent = LD_extent(
    X_location(b_start, geometry_a),
    X_location(b_end, geometry_a),
  );

  var best_depth = depth_b[b_offset];
  var best_data = data_b[b_offset];
  let a_width = i32(geometry_a.width);
  let a_height = i32(geometry_a.height);
  for (var a_y = i32(a_extent.y); a_y < i32(a_extent.w); a_y++) {
    for (var a_x = i32(a_extent.x); a_x < i32(a_extent.z); a_x++) {
      if (a_x < 0 || a_y < 0 || a_x >= a_width || a_y >= a_height) {
        continue;
      }
      let a_location = L_location(vec3<u32>(u32(a_x), u32(a_y), 0u), geometry_a);
      if (a_location.valid) {
        let a_depth = depth_a[a_location.offset];
        if (a_depth < best_depth) {
          best_depth = a_depth;
          best_data = data_a[a_location.offset];
        }
      }
    }
  }
  output_depth[b_offset] = best_depth;
  output_data[b_offset] = best_data;
}
