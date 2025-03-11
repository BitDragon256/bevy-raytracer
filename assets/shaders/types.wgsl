struct RaytracingCamera {
    bounces: u32,
    samples: u32,

    position: vec3<f32>,
    direction: vec3<f32>,
    up: vec3<f32>,

    aspect: f32,
    near: f32,
    far: f32,
    fov: f32,
}

struct NEMesh {
    vertex_offset: u32,
    index_start: u32,
    index_count: u32,
}
struct NEVertex {
    position: vec3<f32>,
    cell_ref: CellRef,
}

struct CellRef {
    index: u32,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct HitInfo {
    hit: bool,
    intersection: vec3<f32>,
}