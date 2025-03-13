#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

//#import "shaders/intersection.wgsl"::intersect_triangle
// INTERSECTION.wgsl

fn intersect_triangle(ray: Ray, a: vec3f, b: vec3f, c: vec3f) -> HitInfo {
	var intersection = HitInfo();

    let edge1 = b - a;
	let edge2 = c - a;

	let ray_cross_edge2 = cross(ray.direction, edge2);
	let det = dot(edge1, ray_cross_edge2);

	if det > EPSILON && det < EPSILON {
		return intersection; // This ray is parallel to this triangle.
	}

	let inv_det = 1.0 / det;
	let s = ray.origin - a;
	let u = inv_det * dot(s, ray_cross_edge2);
	if u < 0.0 || u > 1.0 {
		return intersection;
	}

	let s_cross_edge1 = cross(s, edge1);
	let v = inv_det * dot(ray.direction, s_cross_edge1);
	if v < 0.0 || u + v > 1.0 {
		return intersection;
	}
	// At this stage we can compute t to find out where the intersection point is on the line.
	let t = inv_det * dot(edge2, s_cross_edge1);

	if t > EPSILON { // ray intersection
		intersection.intersection = ray.origin + ray.direction * t;
		intersection.hit = true;
	}

    intersection.normal = normalize(cross(edge1, edge2));
    intersection.distance = t;

	return intersection;
}

// #import "shaders/types.wgsl"::{
//     RaytracingCamera,
//     NEMesh,
//     NEVertex,
//     CellRef,
//     Ray,
//     HitInfo,
//     Material,
//     BSDFContext,
// }
// TYPES.wgsl

struct RaytracingCamera {
    bounces: u32,
    samples: u32,

    position: vec3f,
    direction: vec3f,
    up: vec3f,

    aspect: f32,
    near: f32,
    far: f32,
    fov: f32,
}

struct NEMesh {
    vertex_offset: u32,
    face_start: u32,
    face_count: u32,
    material_offset: u32,
}
struct NEVertex {
    position: vec3f,
    cell_ref: CellRef,
}

struct NETriFace {
    a: u32, b: u32, c: u32,
    material_index: u32,
}

struct CellRef {
    index: u32,
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
}

struct HitInfo {
    hit: bool,
    intersection: vec3f,
    normal: vec3f,
    distance: f32,
    face_index: u32,
    material_index: u32,
}

struct Material {
    bsdf: u32,
    radiance: vec3f, // TODO seperate area light
    albedo: vec3f,
    specular: vec3f,
    exponent: f32,
}

struct BSDFContext {
    incident_dir: vec3f,
    outgoing_dir: vec3f,
    relative_refractive_index: f32,
    uv: vec2f,
    color: vec3f,
}

// #import "shaders/const.wgsl"::{INV_PI, INV_TWOPI}
// CONST.wgsl
const PI: f32 = 3.14159265358979323846;
const INV_PI: f32 = 0.31830988618379067154;
const INV_TWOPI: f32 = 0.15915494309189533577;
const EPSILON: f32 = 0.000001;
const INF: f32 = 3.40282347e+38;
const RAD2DEG: f32 = 180.0 * INV_PI;
const DEG2RAD: f32 = PI / 180.0;

// #import "shaders/warp.wgsl"::{square_to_cosine_hemisphere_pdf, square_to_cosine_hemisphere, square_to_uniform_disk}
// WARP.wgsl
fn square_to_cosine_hemisphere_pdf(v: vec3f) -> f32 {
    return max(0.0, v.z) * INV_PI;
}
fn square_to_uniform_disk(v: vec2f) -> vec2f {
    let theta = v.x * PI * 2.0;
    let r = sqrt(v.y);

    // TODO find more efficient way to calculate this
    return vec2f(cos(theta), sin(theta)) * r;
}
fn square_to_cosine_hemisphere(v: vec2f) -> vec3f {
    let disk = square_to_uniform_disk(v);
    return vec3f(disk.x, disk.y, sqrt(1.0 - square_length(disk)));
}

// RAYTRACE.wgsl

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

@group(0) @binding(2) var<uniform> camera: RaytracingCamera;

@group(1) @binding(0) var<storage, read> mesh_buffer: array<NEMesh>;
@group(1) @binding(1) var<storage, read> vertex_buffer: array<NEVertex>;
@group(1) @binding(2) var<storage, read> tri_face_buffer: array<NETriFace>;
@group(1) @binding(3) var<storage, read> material_buffer: array<Material>;

fn intersect_mesh(ray: Ray, mesh: NEMesh) -> HitInfo {
    var nearest_hit_info = HitInfo();
    nearest_hit_info.distance = INF;

    for (var face_index = mesh.face_start; face_index < mesh.face_start + mesh.face_count; face_index++) {
        let face = tri_face_buffer[face_index];
        let a = vertex_buffer[face.a + mesh.vertex_offset].position;
        let b = vertex_buffer[face.b + mesh.vertex_offset].position;
        let c = vertex_buffer[face.c + mesh.vertex_offset].position;
        let tri_hit_info = intersect_triangle(ray, a, b, c);

        if tri_hit_info.hit && tri_hit_info.distance < nearest_hit_info.distance {
            nearest_hit_info = tri_hit_info;
            nearest_hit_info.face_index = face_index;
            nearest_hit_info.material_index = face.material_index + mesh.material_offset;
        }
    }
    return nearest_hit_info;
}

fn ray_from_uv(uv: vec2f) -> Ray {
    let delta = vec2f(
        uv.x * 2.0 - 1.0,
        1.0 - uv.y * 2.0,
    );
    let right = cross(camera.direction, camera.up);
    let scale = tan(camera.fov * DEG2RAD * 0.5);
    let dir = normalize(camera.direction + (delta.x * camera.aspect * scale * right) + (delta.y * scale * camera.up));
    return Ray(camera.position, dir);
}

fn reflect(v: vec3f) -> vec3f {
    return vec3f(-v.x, -v.y, v.z);
}
fn cos_theta(v: vec3f) -> f32 {
    return v.z;
}
fn square_length(v: vec2f) -> f32 {
    return v.x * v.x + v.y * v.y;
}

struct Frame {
    u: vec3f, // tangent 1
    v: vec3f, // tangent 2
    n: vec3f, // normal
}

fn create_frame(n: vec3f) -> Frame {
    var f = Frame();
    f.n = n;

    if abs(n.x) > abs(n.y) {
        f.v = vec3f(n.z, 0.0, -n.x) / length(n.xz);
    } else {
        f.v = vec3f(0.0, n.z, -n.y) / length(n.yz);
    }
    f.u = cross(f.v, n);

    return f;
}

// converts v from world space to the local frame f
fn to_local(v: vec3f, f: Frame) -> vec3f {
    return vec3f(
        dot(v, f.u),
        dot(v, f.v),
        dot(v, f.n)
    );
}
// converts v from the local frame f to world space
fn to_world(v: vec3f, f: Frame) -> vec3f {
    return f.u * v.x + f.v * v.y + f.n * v.z;
}

fn phong_eval(context: BSDFContext, material: Material) -> vec3f {
    let alpha = dot(context.outgoing_dir, reflect(context.incident_dir));
    var spec_pdf = 0.0;
    if alpha > 0.0 {
        spec_pdf = pow(alpha, material.exponent) * (material.exponent + 1.0) * INV_TWOPI;
    }
    return material.albedo * INV_PI + material.specular * spec_pdf;
}
fn phong_pdf(context: BSDFContext, material: Material) -> f32 {
    let alpha = dot(context.outgoing_dir, reflect(context.incident_dir));
    var spec_pdf = 0.0;
    if alpha > 0.0 {
        spec_pdf = pow(alpha, material.exponent) * (material.exponent + 1.0) * INV_TWOPI;
    }
    let diff_pdf = square_to_cosine_hemisphere_pdf(context.outgoing_dir);
    // TODO real spec sampling rate
    let spec_sampling_rate = 1.0;
    return diff_pdf + (spec_pdf - diff_pdf) / spec_sampling_rate;
}
fn phong_sample(incident_dir: vec3f, rng_sample: vec2f, material: Material) -> BSDFContext {
    var context = BSDFContext();
    context.incident_dir = incident_dir;

    if cos_theta(incident_dir) <= 0.0 {
        context.color = vec3f(0.0);
        return context;
    }
    // TODO real spec sampling rate
    let spec_sampling_rate = 1.0;
    let sample_spec = rng_sample.y < spec_sampling_rate;
    if sample_spec {
        let spec_sample = vec2f(rng_sample.x, rng_sample.y / spec_sampling_rate);
        let disk_sample = square_to_uniform_disk(spec_sample);
        let radius_sample = square_length(disk_sample);
        if radius_sample > 0.0 {
            let cos_theta = pow(radius_sample, 1.0 / (material.exponent + 1.0));
            let sin_theta_div_radius = sqrt((1.0 - cos_theta * cos_theta) / radius_sample);
            context.outgoing_dir = vec3f( sin_theta_div_radius * disk_sample.x, sin_theta_div_radius * disk_sample.y, cos_theta );

            let reflection_frame = create_frame(reflect(context.incident_dir));
            context.outgoing_dir = to_world(context.outgoing_dir, reflection_frame);
        } else {
            context.outgoing_dir = reflect(incident_dir);
        }
    } else {
        let diff_sample = vec2f(rng_sample.x, (rng_sample.y - spec_sampling_rate) / (1.0 - spec_sampling_rate));
        context.outgoing_dir = square_to_cosine_hemisphere(diff_sample);
    }

    context.relative_refractive_index = 1.0;
    if cos_theta(context.outgoing_dir) <= 0 {
        context.color = vec3f(0.0);
        return context;
    }

    context.color = phong_eval(context, material) * cos_theta(context.outgoing_dir) / phong_pdf(context, material);

    return context;
}

fn diffuse_eval(context: BSDFContext, material: Material) -> vec3f {
    if cos_theta(context.incident_dir) <= 0 || cos_theta(context.outgoing_dir) <= 0 {
        return vec3f(0.0);
    }
    return material.albedo * INV_PI;
}
fn diffuse_pdf(context: BSDFContext, material: Material) -> f32 {
    if cos_theta(context.incident_dir) <= 0 || cos_theta(context.outgoing_dir) <= 0 {
        return 0.0;
    }
    return INV_PI * cos_theta(context.outgoing_dir);
}
fn diffuse_sample(incident_dir: vec3f, rng_sample: vec2f, material: Material) -> BSDFContext {
    var context = BSDFContext();
    context.incident_dir = incident_dir;

    if cos_theta(incident_dir) <= 0.0 {
        context.color = vec3f(0.0);
        return context;
    }

    context.outgoing_dir = square_to_cosine_hemisphere(rng_sample);
    context.color = material.albedo;
    context.relative_refractive_index = 1.0;

    return context;
}

fn sample_bsdf(incident_dir: vec3f, rng_sample: vec2f, material: Material) -> BSDFContext {
    if material.bsdf == 0 {
        return diffuse_sample(incident_dir, rng_sample, material);
    } else if material.bsdf == 1 {
        return phong_sample(incident_dir, rng_sample, material);
    }
    return BSDFContext();
}

fn trace_ray(ray: Ray) -> HitInfo {
    var nearest_hit_info = HitInfo();
    nearest_hit_info.distance = INF;

    // move origin slightly to eliminate rounding errors
    let mod_ray = Ray(ray.origin + ray.direction * 0.01, ray.direction);

    for (var mesh_index: u32 = 0; mesh_index < arrayLength(&mesh_buffer); mesh_index += 1u) {
        let mesh = mesh_buffer[mesh_index];
        let mesh_hit_info = intersect_mesh(mod_ray, mesh);
        if mesh_hit_info.hit && mesh_hit_info.distance < nearest_hit_info.distance {
            nearest_hit_info = mesh_hit_info;
        }
    }
    return nearest_hit_info;
}

// RNG from https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application
// Example 37-4
struct RngState {
    z1: u32,
    z2: u32,
    z3: u32,
    z4: u32,
}

fn TausStep(z: ptr<function, u32>, s1: u32, s2: u32, s3: u32, m: u32) -> u32 {
    let b = ((*z << s1) ^ *z) >> s2;
    *z = ((*z & m) << s3) ^ b;
    return *z;
}
fn LCGStep(z: ptr<function, u32>, a: u32, c: u32) -> u32 {
    *z = a * *z + c;
    return *z;
}
fn HybridTaus(state: ptr<function, RngState>) -> f32 {
    // the state has to be unwrapped, as wgsl can't handle pointers to structs for whatever reason
    var z1 = (*state).z1;
    var z2 = (*state).z2;
    var z3 = (*state).z3;
    var z4 = (*state).z4;
    let res = 2.3283064365387e-10 * f32(              // Periods
        TausStep(&z1, 13u, 19u, 12u, 4294967294u) ^  // p1=2^31-1
        TausStep(&z2, 2u, 25u, 4u, 4294967288u) ^    // p2=2^30-1
        TausStep(&z3, 3u, 11u, 17u, 4294967280u) ^   // p3=2^28-1
        LCGStep(&z4, 1664525u, 1013904223u)        // p4=2^32
    );
    (*state).z1 = z1;
    (*state).z2 = z2;
    (*state).z3 = z3;
    (*state).z4 = z4;
    return res;
}

fn next_random(state: ptr<function, RngState>) -> f32 {
    return HybridTaus(state);
}
fn next_random_2d(state: ptr<function, RngState>) -> vec2f {
    return vec2f(next_random(state), next_random(state));
}
fn uv_to_rng_state(uv: vec2f) -> RngState {
    return RngState(
        bitcast<u32>(fract(sin(dot(uv.xy, vec2f(12.9898, 78.2338))) * 43758.5453123)),
        bitcast<u32>(fract(sin(dot(camera.position.xz, vec2f(91.4435, 42.1895))) * 15512.5037034)),
        bitcast<u32>(fract(sin(dot(camera.position.yz, vec2f(83.9575, 67.0150))) * 32739.7101972)),
        bitcast<u32>(fract(sin(dot(uv.yx, vec2f(39.6784, 53.7602))) * 77672.1025434)),
    );
}

fn is_emitter(material: Material) -> bool {
    return dot(material.radiance, material.radiance) != 0.0;
}
fn eval_emitter(incident_dir: vec3f, material: Material) -> vec3f {
    return material.radiance * cos_theta(incident_dir);
}

fn eval_bsdf_path(in_ray: Ray, rng_state: ptr<function, RngState>) -> vec3f {
    var throughput = vec3f(1.0);
    var radiance = vec3f(0.0);
    var ray = in_ray;

    for (var bounce = 0u; bounce < camera.bounces; bounce++) {
        let hit_info = trace_ray(ray);
        if !hit_info.hit {
            break;
        }

        let local_frame = create_frame(hit_info.normal);
        let incident_dir = to_local(-ray.direction, local_frame);

        let material = material_buffer[hit_info.material_index];

        let bsdf_context = sample_bsdf(incident_dir, next_random_2d(rng_state), material);

        if is_emitter(material) {
            radiance += eval_emitter(incident_dir, material) * throughput;
        }

        throughput *= bsdf_context.color;
        ray = Ray(hit_info.intersection, to_world(bsdf_context.outgoing_dir, local_frame));

        // return ray.direction * 0.5 + vec3f(0.5);
    }

    return radiance;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4f {
    var rng_state = uv_to_rng_state(in.uv);
    var color = vec3f(0.0);
    for (var sample_index = 0u; sample_index < camera.samples; sample_index++) {
        color += eval_bsdf_path(ray_from_uv(in.uv), &rng_state);
    }
    return vec4f(color / f32(camera.samples), 1.0);
}