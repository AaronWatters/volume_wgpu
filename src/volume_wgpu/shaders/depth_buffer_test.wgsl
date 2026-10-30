@group(0) @binding(0) var<storage, read_write> output_data: array<f32>;

@compute @workgroup_size(1)
fn depth_buffer_extent_test() {
    let geometry = DepthBufferGeometry(100.0, 100.0, 3.0, 4.0, 1.0, 1.0);
    let extent = L_extent(
        vec3<f32>(4.0, 5.0, 0.0),
        vec3<f32>(6.0, 8.0, 0.0),
        geometry,
    );
    output_data[0] = f32(extent.x);
    output_data[1] = f32(extent.y);
    output_data[2] = f32(extent.z);
    output_data[3] = f32(extent.w);
};
