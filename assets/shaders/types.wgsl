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

struct Material {
    bsdf: u32,
    emitter: u32,
    diffuse: vec3<f32>,
    specular: vec3<f32>,
    exponent: f32,
}

struct BSDFContext {
    incident_dir: vec3<f32>,
    outgoing_dir: vec3<f32>,
    relative_refractive_index: f32,
    uv: vec2<f32>,
    color: vec3<f32>,
}

struct AreaEmitter {
    surface_area: f32,
    radiance: f32,
}