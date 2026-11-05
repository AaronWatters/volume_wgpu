
// Shader suffix for max value computation:
// Assume prefix depth_buffer.wgsl and volume.wgsl
//
// Computation projects max value from the volume onto a depth buffer

// input volume:
@group(0) @binding(0) var<storage, read_write> input_data: array<f32>;
@group(0) @binding(1) var<uniform> input_geometry: VolumeGeometry;

// output depth buffer:
// buffers must be atomic for concurrent updates, bitcast to f32.
@group(1) @binding(0) var<storage, read_write> output_data: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> output_depth: array<atomic<u32>>;
@group(1) @binding(2) var<uniform> output_geometry: DepthBufferGeometry;


@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_number = global_id.x;

    /*
    // debugging: dump shape.
    if (row_number < 3u) {
        output_data[row_number] = bitcast<u32>(f32(input_geometry.shape[row_number]));
        //output_data[row_number] = bitcast<u32>(f32(111));
        return;
    }
    */
    
    let nrows = nrows(input_geometry);
    if (row_number >= nrows) {
        //output_data[row_number] = bitcast<u32>(f32(nrows));
        return;
    }
    
    /*
    // debugging:
    let length = arrayLength(&output_data);
    if (row_number < length) {
        // debugging:
        //let f32_value = f32(row_number);
        let f32_value = f32(nrows);
        output_data[row_number] = bitcast<u32>(f32_value);
    }
    */
    
    let row_size = row_size(input_geometry);
    let start_offset = row_number * row_size;
    for (var dk = 0u; dk < row_size; dk++) {
        let offset = start_offset + dk;
        let value = input_data[offset];
        let I_ = offset_to_I(input_geometry, offset);
        let extent = voxel_extent(input_geometry, I_);
        let mins = extent.mins;
        let maxs = extent.maxs;
        let L_min = X_to_L(mins, output_geometry);
        let L_max = X_to_L(maxs, output_geometry);
        // use floor and ceil to determine the integer bounds in the output space
        let minL = u32(floor(L_min.x));
        let maxL = u32(ceil(L_max.x));
        let minM = u32(floor(L_min.y));
        let maxM = u32(ceil(L_max.y));
        let z = mins.z;
        for (var L = minL; L <= maxL; L++) {
            for (var M = minM; M <= maxM; M++) {
                let L_ = vec3<u32>(L, M, 0u);
                let L_loc = L_location(L_, output_geometry);
                if (L_loc.valid) {
                    let index = L_loc.offset;
                    let current_data = bitcast<f32>(atomicLoad(&output_data[index]));
                    if (value > current_data) {
                        // Atomically update the max value and depth using atomicStore and atomicMax
                        //output_data[index] = value;
                        //output_depth[index] = z;
                        atomicStore(&output_data[index], bitcast<u32>(value));
                        atomicStore(&output_depth[index], bitcast<u32>(z));
                    }
                }
            }
        }
    }
    
}