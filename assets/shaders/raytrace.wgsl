#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

#import "shaders/random.wgsl"::{rngNextFloat, randomUnitVec3}
#import "shaders/const.wgsl"::{PI, INF}

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

@group(0) @binding(2) var<uniform> camera: RaytracingCamera;

// @group(1) @binding(0) var mesh_buffer: array<NEMesh>;

struct RaytracingCamera {
    bounces: u32,
    samples: u32,

    pos: vec3<f32>,
    dir: vec3<f32>,
    up: vec3<f32>,

    aspect: f32,
    near: f32,
    far: f32,
    fov: f32,
}

struct NEMesh {
    vertex_offset: u32,
    index_start: u32,
}
struct NEVertex {
    pos: vec3<f32>,
    cell_ref: CellRef,
}

struct CellRef {
    index: u32,
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}