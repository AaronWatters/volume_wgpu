
// WSGL structure for depth buffer geometry with operations
struct DepthBufferGeometry {
    width: f32,
    height: f32,
    x0: f32,
    y0: f32,
    dx: f32,
    dy: f32,
};

struct DepthBufferLocation {
    valid: bool,
    offset: u32,
    L: vec2<u32>,
    X: vec3<f32>,
    z: f32,
    data: f32,
};

// Analogous operations to reference/depth_buffer.py
fn X_to_L(X_: vec3<f32>, geometry: DepthBufferGeometry) -> vec2<u32> {
    let L = u32((X_.x - geometry.x0) / geometry.dx);
    let M = u32((X_.y - geometry.y0) / geometry.dy);
    let Z = u32(X_.z);
    return vec2<u32>(L, M);
};

fn L_to_X(L_: vec2<u32>, geometry: DepthBufferGeometry) -> vec3<f32> {
    let X = geometry.x0 + f32(L_.x) * geometry.dx;
    let Y = geometry.y0 + f32(L_.y) * geometry.dy;
    let Z = 0.0; // Assuming Z is 0 for 2D depth buffer
    return vec3<f32>(X, Y, Z);
};

fn L_valid(L_: vec2<u32>, geometry: DepthBufferGeometry) -> bool {
    let width = u32(geometry.width);
    let height = u32(geometry.height);
    return L_.x < width && L_.y < height && L_.x >= 0u && L_.y >= 0u;
};

fn L_offset(L_: vec2<u32>, geometry: DepthBufferGeometry) -> u32 {
    let width = u32(geometry.width);
    return L_.y * width + L_.x;
};

fn offset_to_L(offset: u32, geometry: DepthBufferGeometry) -> vec2<u32> {
    let width = u32(geometry.width);
    let height = u32(geometry.height);
    let L = offset % width;
    let M = offset / width;
    return vec2<u32>(L, M);
};

fn L_location(L_: vec3<u32>, geometry: DepthBufferGeometry) -> DepthBufferLocation {
    let valid = L_valid(L_.xy, geometry);
    let offset = L_offset(L_.xy, geometry);
    let X = L_to_X(L_.xy, geometry);
    let data = 0.0;
    let z = 0.0;
    return DepthBufferLocation(valid, offset, L_.xy, X, z, data);
};

fn X_location(X_: vec3<f32>, geometry: DepthBufferGeometry) -> DepthBufferLocation {
    let L = X_to_L(X_, geometry);
    let valid = L_valid(L, geometry);
    let offset = L_offset(L, geometry);
    let z = 0.0;
    let data = 0.0;
    return DepthBufferLocation(valid, offset, L, X_, z, data);
};

fn X_offset(X_: vec3<f32>, geometry: DepthBufferGeometry) -> u32 {
    let L = X_to_L(X_, geometry);
    return L_offset(L, geometry);
};

fn L_extent(X1: vec3<f32>, X2: vec3<f32>, geometry: DepthBufferGeometry) -> vec4<u32> {
    let L1 = X_to_L(X1, geometry);
    let L2 = X_to_L(X2, geometry);
    let L_min = vec2<u32>(min(L1.x, L2.x), min(L1.y, L2.y));
    let L_max = vec2<u32>(max(L1.x, L2.x), max(L1.y, L2.y));
    return vec4<u32>(L_min.x, L_min.y, L_max.x, L_max.y);
};
