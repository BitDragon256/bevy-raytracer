#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

#import "shaders/intersection.wgsl"::intersect_triangle
#import "shaders/types.wgsl"::{
    RaytracingCamera,
    NEMesh,
    NEVertex,
    CellRef,
    Ray,
    HitInfo
}

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

@group(0) @binding(2) var<uniform> camera: RaytracingCamera;

@group(1) @binding(0) var<storage, read> mesh_buffer: array<NEMesh>;
@group(1) @binding(1) var<storage, read> vertex_buffer: array<NEVertex>;
@group(1) @binding(2) var<storage, read> index_buffer: array<u32>;

fn intersect_mesh(ray: Ray, mesh: NEMesh) -> HitInfo {
    for (var index = mesh.index_start; index < mesh.index_start + mesh.index_count; index += 3u) {
        let a = vertex_buffer[index_buffer[index + 0] + mesh.vertex_offset].position;
        let b = vertex_buffer[index_buffer[index + 1] + mesh.vertex_offset].position;
        let c = vertex_buffer[index_buffer[index + 2] + mesh.vertex_offset].position;
        let tri_hit_info = intersect_triangle(ray, a, b, c);

        if tri_hit_info.hit {
            return tri_hit_info;
        }
    }
    return HitInfo();
}

fn ray_from_uv(uv: vec2<f32>) -> Ray {
    let delta = vec2<f32>(
        uv.x * 2.0 - 1.0,
        1.0 - uv.y * 2.0,
    );
    let right = cross(camera.direction, camera.up);
    let dir = normalize(camera.direction + (delta.x * camera.aspect * right) + (delta.y * camera.up));
    return Ray(camera.position, dir);
}

fn trace_ray(ray: Ray) -> HitInfo {
    for (var mesh_index: u32 = 0; mesh_index < arrayLength(&mesh_buffer); mesh_index += 1u) {
        let mesh = mesh_buffer[mesh_index];
        var mesh_hit_info = intersect_mesh(ray, mesh);
        if mesh_hit_info.hit {
            return mesh_hit_info;
        }
    }
    return HitInfo();
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    if trace_ray(ray_from_uv(in.uv)).hit {
        return vec4<f32>(0.0, 1.0, 0.0, 1.0);
    }

    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}