

// WGSL prefix with volume geometry logic
// analogous to the reference implementation in Python reference/volume.py

struct VolumeGeometry {
    transform: mat4x4<f32>,
    inverse: mat4x4<f32>,
    // shape as f32 is sufficient
    shape: vec3<f32>,
};

struct VolumeLocation {
    I_: vec3<f32>,
    valid: bool,
    offset: u32,
    X_: vec3<f32>,
};

fn nrows(geometry: VolumeGeometry) -> u32 {
    let Ni = u32(geometry.shape.x);
    let Nj = u32(geometry.shape.y);
    return Ni * Nj;
};

fn row_size(geometry: VolumeGeometry) -> u32 {
    let Nk = u32(geometry.shape.z);
    return Nk;
};

// set_M in python

fn I_to_X(geometry: VolumeGeometry, I_: vec3<f32>) -> vec3<f32> {
    let I_1 = vec4<f32>(I_, 1.0);
    let X_1 = geometry.transform * I_1;
    return X_1.xyz;
};

fn X_to_I(geometry: VolumeGeometry, X_: vec3<f32>) -> vec3<f32> {
    let X_1 = vec4<f32>(X_, 1.0);
    let I_1 = geometry.inverse * X_1;
    return I_1.xyz;
};

fn I_to_offset(geometry: VolumeGeometry, I_: vec3<f32>) -> u32 {
    let Ni = u32(geometry.shape.x);
    let Nj = u32(geometry.shape.y);
    let Nk = u32(geometry.shape.z);
    let i = u32(I_.x);
    let j = u32(I_.y);
    let k = u32(I_.z);
    return (i * Nj + j) * Nk + k;
};

fn offset_to_I(geometry: VolumeGeometry, offset: u32) -> vec3<f32> {
    let Ni = u32(geometry.shape.x);
    let Nj = u32(geometry.shape.y);
    let Nk = u32(geometry.shape.z);
    let k = offset % Nk;
    let j = (offset / Nk) % Nj;
    let i = offset / (Nj * Nk);
    return vec3<f32>(f32(i), f32(j), f32(k));
};

fn X_to_offset(geometry: VolumeGeometry, X_: vec3<f32>) -> u32 {
    let I_ = X_to_I(geometry, X_);
    return I_to_offset(geometry, I_);
};

fn offset_to_X(geometry: VolumeGeometry, offset: u32) -> vec3<f32> {
    let I_ = offset_to_I(geometry, offset);
    return I_to_X(geometry, I_);
};

fn valid_I(geometry: VolumeGeometry, I_: vec3<f32>) -> bool {
    let Ni = u32(geometry.shape.x);
    let Nj = u32(geometry.shape.y);
    let Nk = u32(geometry.shape.z);
    let i = u32(I_.x);
    let j = u32(I_.y);
    let k = u32(I_.z);
    return i < Ni && j < Nj && k < Nk;
};

struct Vec6 {
    mins: vec3<f32>,
    maxs: vec3<f32>,
}

// return min and max (x,y,z) of the voxel at index I_
fn voxel_extent(geometry: VolumeGeometry, I_: vec3<f32>) -> Vec6 {
    let X_ = I_to_X(geometry, I_);
    var mins = X_;
    var maxs = X_;
    //let [i, j, k] = I_;
    for (var di = 0; di < 2; di++) {
        for (var dj = 0; dj < 2; dj++) {
            for (var dk = 0; dk < 2; dk++) {
                let dI = vec3<f32>(f32(di), f32(dj), f32(dk));
                let IdI_ = I_ + dI;
                let X_ = I_to_X(geometry, IdI_);
                mins = min(mins, X_);
                maxs = max(maxs, X_);
            }
        }
    }
    return Vec6(mins, maxs);
};
