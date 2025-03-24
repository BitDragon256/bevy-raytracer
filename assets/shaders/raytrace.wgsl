#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

//#import "shaders/intersection.wgsl"::intersect_triangle
// INTERSECTION.wgsl

fn intersect_triangle(ray: Ray, a: vec3f, b: vec3f, c: vec3f) -> HitInfo {
	var intersection = HitInfo();
    intersection.distance = INF;

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

	if t > ray.min && t < ray.max { // ray intersection
		intersection.intersection = ray.origin + ray.direction * t;
		intersection.hit = true;
	}

    intersection.distance = t;
    intersection.uv = vec2f(u, v);

	return intersection;
}

fn intersect_aabb(ray_origin: vec3f, ray_inv_dir: vec3f, ray_max: f32, box_min: vec3f, box_max: vec3f) -> f32 {
    let t1 = (box_min - ray_origin) * ray_inv_dir;
    let t2 = (box_max - ray_origin) * ray_inv_dir;

    let tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));

    return select(INF, tmin, tmax >= tmin && tmax > 0 && tmin <= ray_max);
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
    min_bounces: u32,
    max_bounces: u32,
    bounce_probability: f32,

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
    face_offset: u32,
    bvh_root: u32,
    bvh_size: u32,
    material_offset: u32,
    transform_index: u32,
    surface_area: f32,
    flattened_bvh: u32,
}
struct NEVertex {
    position: vec3f,
    normal: vec3f,
    texture: vec3f,
    cell_ref: CellRef,
}

struct NETriFace {
    a: u32, b: u32, c: u32,
    material_index: u32,
    bvh_index: u32,
}

struct BvhNode {
    min: vec3f,
    max: vec3f,
    entry_index: u32,
    exit_index: u32,
    shape_index: u32,
}

struct CellRef {
    index: u32,
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
    min: f32,
    max: f32,
}

fn make_inf_ray(origin: vec3f, direction: vec3f) -> Ray {
    return make_ray(origin, direction, INF);
}
fn make_ray(origin: vec3f, direction: vec3f, max: f32) -> Ray {
    return Ray(
        origin,
        direction,
        EPSILON,
        max,
    );
}

struct HitInfo {
    hit: bool,
    intersection: vec3f,
    normal: vec3f,
    distance: f32,
    uv: vec2f,

    face_index: u32,
    material_index: u32,
    mesh_index: u32,
    test_count: f32,
}

struct Material {
    bsdf: u32,
    radiance: vec3f, // TODO seperate area light
    albedo: vec3f,
    specular: vec3f,
    exponent: f32,
    extIOR: f32, // IOR == index of refraction
    intIOR: f32,
    eta: vec3f,
    k: vec3f,
}

struct BSDFContext {
    incident_dir: vec3f,
    outgoing_dir: vec3f,
    relative_refractive_index: f32,
    uv: vec2f,
    color: vec3f,
}

struct Transform {
    translate: vec3f,
    scale: vec3f,
    rotate: mat3x3f,
    inv_rotate: mat3x3f,
}

struct Light {
    face_index: u32,
    mesh_index: u32,
}

struct Triangle {
    a: NEVertex,
    b: NEVertex,
    c: NEVertex,
}

fn get_triangle(face: NETriFace, mesh: NEMesh, transform: Transform) -> Triangle {
    var tri = Triangle(
        vertex_buffer[mesh.vertex_offset + face.a],
        vertex_buffer[mesh.vertex_offset + face.b],
        vertex_buffer[mesh.vertex_offset + face.c],
    );

    tri.a.position = apply_transform(transform, tri.a.position);
    tri.b.position = apply_transform(transform, tri.b.position);
    tri.c.position = apply_transform(transform, tri.c.position);

    tri.a.normal = apply_scale_rotation(transform, tri.a.normal);
    tri.b.normal = apply_scale_rotation(transform, tri.b.normal);
    tri.c.normal = apply_scale_rotation(transform, tri.c.normal);
    
    return tri;
}

// #import "shaders/const.wgsl"::{INV_PI, INV_TWOPI}
// CONST.wgsl
const PI: f32 = 3.14159265358979323846;
const INV_PI: f32 = 0.31830988618379067154;
const INV_TWOPI: f32 = 0.15915494309189533577;
const EPSILON: f32 = 0.000001;
const INF: f32 = 3.40282347e+38;
const U32_MAX: u32 = 4294967295u;
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
fn square_to_uniform_triangle(v: vec2f) -> vec2f {
    if v.y >= v.x {
        return -v + vec2f(1.0);
    }
    return v;
}
fn uniform_triangle_to_triangle(v: vec2f, a: vec3f, b: vec3f, c: vec3f) -> vec3f {
    let sqrtU = sqrt(v.x);
    let barycentric = vec3f(1.0 - sqrtU, sqrtU * (1.0 - v.y), sqrtU * v.y);
    return barycentric.x * a + barycentric.y * b + barycentric.z * c;
}

// RAYTRACE.wgsl

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

@group(0) @binding(2) var<uniform> camera: RaytracingCamera;

@group(1) @binding(0) var<storage, read> mesh_buffer: array<NEMesh>;
@group(1) @binding(1) var<storage, read_write> llas_buffer: array<BvhNode>;
@group(1) @binding(2) var<storage, read> vertex_buffer: array<NEVertex>;
@group(1) @binding(3) var<storage, read> tri_face_buffer: array<NETriFace>;
@group(1) @binding(4) var<storage, read> material_buffer: array<Material>;
@group(1) @binding(5) var<storage, read> transform_buffer: array<Transform>;
@group(1) @binding(6) var<storage, read> light_buffer: array<Light>;

fn apply_transform(transform: Transform, v: vec3f) -> vec3f {
    return transform.translate + transform.scale * (transform.rotate * v);
}
fn apply_inv_transform(transform: Transform, v: vec3f) -> vec3f {
    return transform.inv_rotate * ((v - transform.translate) / transform.scale);
}
fn apply_scale_rotation(transform: Transform, v: vec3f) -> vec3f {
    return transform.scale * (transform.rotate * v);
}
fn apply_rotation(transform: Transform, v: vec3f) -> vec3f {
    return transform.rotate * v;
}
fn apply_inv_rotation(transform: Transform, v: vec3f) -> vec3f {
    return transform.inv_rotate * v;
}
fn apply_scale_translate(transform: Transform, v: vec3f) -> vec3f {
    return transform.translate + transform.scale * v;
}
fn apply_inv_scale_translate(transform: Transform, v: vec3f) -> vec3f {
    return (v - transform.translate) / transform.scale;
}

fn intersect_mesh_stack(ray: Ray, mesh: NEMesh) -> HitInfo {
    var test_count = 0.0;

    var nearest_hit_info = HitInfo();
    nearest_hit_info.distance = INF;

    let bvh_offset = mesh.bvh_root;
    let transform = transform_buffer[mesh.transform_index];

    var mod_ray = ray;

    let inv_ray_dir = 1.0 / apply_inv_rotation(transform, ray.direction);
    let aabb_ray_origin = apply_inv_transform(transform, ray.origin);

    var stack = array<u32, 128>();
    var stack_index = 1u;

    stack[0] = 0u;

    while stack_index > 0u && test_count < 500.0 {
        stack_index--;
        let bvh_index = stack[stack_index];

        if bvh_index >= mesh.bvh_size {
            break;
        }

        let bvh_node = llas_buffer[bvh_offset + bvh_index];

        if bvh_node.shape_index != U32_MAX { // leaf node
            let face = tri_face_buffer[bvh_node.shape_index + mesh.face_offset];

            let tri = get_triangle(face, mesh, transform);
            let tri_hit_info = intersect_triangle(mod_ray, tri.a.position, tri.b.position, tri.c.position);

            if tri_hit_info.hit && tri_hit_info.distance < nearest_hit_info.distance {
                nearest_hit_info = tri_hit_info;
                nearest_hit_info.face_index = bvh_node.shape_index + mesh.face_offset;
                nearest_hit_info.material_index = face.material_index + mesh.material_offset;

                mod_ray.max = tri_hit_info.distance;
            }
        } else {
            test_count += 2.0;

            let left_child = llas_buffer[bvh_offset + bvh_node.entry_index];
            let right_child = llas_buffer[bvh_offset + bvh_node.exit_index];

            let l = intersect_aabb(
                aabb_ray_origin,
                inv_ray_dir,
                INF, // TODO mod_ray.max - mod_ray.min,
                left_child.min,
                left_child.max
            );

            let r = intersect_aabb(
                aabb_ray_origin,
                inv_ray_dir,
                INF, // TODO mod_ray.max - mod_ray.min,
                right_child.min,
                right_child.max
            );

            if l < r { // left is nearer
                if r != INF {
                    stack[stack_index] = bvh_node.exit_index;
                    stack_index++;
                }
                stack[stack_index] = bvh_node.entry_index;
                stack_index++;
            } else { // right is nearer
                if l != INF {
                    stack[stack_index] = bvh_node.entry_index;
                    stack_index++;
                }
                if r != INF {
                    stack[stack_index] = bvh_node.exit_index;
                    stack_index++;
                }
            }
        }
    }

    nearest_hit_info.test_count = test_count;
    return nearest_hit_info;
}
fn intersect_mesh(ray: Ray, mesh: NEMesh) -> HitInfo {
    if mesh.flattened_bvh == 0 {
        return intersect_mesh_stack(ray, mesh);
    }

    var test_count = 0.0;

    var nearest_hit_info = HitInfo();
    nearest_hit_info.distance = INF;

    let transform = transform_buffer[mesh.transform_index];

    var mod_ray = ray;

    let inv_ray_dir = 1.0 / apply_inv_rotation(transform, ray.direction);
    let aabb_ray_origin = apply_inv_transform(transform, ray.origin);

    let bvh_offset = mesh.bvh_root;
    var bvh_index = 0u;

    while bvh_index < mesh.bvh_size && test_count < 1000.0 {
        test_count += 1.0;

        let bvh_node = llas_buffer[bvh_index + bvh_offset];

        if bvh_node.entry_index == U32_MAX { // leaf node
            test_count -= 1.0;

            let face = tri_face_buffer[bvh_node.shape_index + mesh.face_offset];

            let tri = get_triangle(face, mesh, transform);
            let tri_hit_info = intersect_triangle(mod_ray, tri.a.position, tri.b.position, tri.c.position);

            if tri_hit_info.hit && tri_hit_info.distance < nearest_hit_info.distance {
                nearest_hit_info = tri_hit_info;
                nearest_hit_info.face_index = bvh_node.shape_index + mesh.face_offset;
                nearest_hit_info.material_index = face.material_index + mesh.material_offset;

                mod_ray.max = tri_hit_info.distance;
            }

            bvh_index = bvh_node.exit_index;
        }
        else {
            let a = intersect_aabb(
                aabb_ray_origin,
                inv_ray_dir,
                INF, // TODO mod_ray.max - mod_ray.min,
                bvh_node.min,
                bvh_node.max
            );
            if a < INF {
                bvh_index = bvh_node.entry_index;
            } else {
                bvh_index = bvh_node.exit_index;
            }
        }
    }

    nearest_hit_info.test_count = test_count;

    return nearest_hit_info;
}

fn trace_ray(ray: Ray) -> HitInfo {
    var nearest_hit_info = HitInfo();
    nearest_hit_info.distance = INF;

    // move origin slightly to eliminate rounding errors
    var mod_ray = ray;
    mod_ray.min = 0.01;

    var test_count = 0.0;

    for (var mesh_index: u32 = 0; mesh_index < arrayLength(&mesh_buffer); mesh_index++) {
        let mesh = mesh_buffer[mesh_index];
        let mesh_hit_info = intersect_mesh(mod_ray, mesh);

        test_count += mesh_hit_info.test_count;

        if mesh_hit_info.hit && mesh_hit_info.distance < nearest_hit_info.distance {
            nearest_hit_info = mesh_hit_info;
            nearest_hit_info.mesh_index = mesh_index;

            mod_ray.max = mesh_hit_info.distance;
        }
    }
    nearest_hit_info.test_count = test_count;

    if nearest_hit_info.hit {
        let face = tri_face_buffer[nearest_hit_info.face_index];
        let mesh = mesh_buffer[nearest_hit_info.mesh_index];
        let transform = transform_buffer[mesh.transform_index];

        let barycentric = vec3f(1.0 - nearest_hit_info.uv.x - nearest_hit_info.uv.y, nearest_hit_info.uv);

        let tri = get_triangle(face, mesh, transform);

        nearest_hit_info.normal = normalize(select(
            cross(tri.b.position - tri.a.position, tri.c.position - tri.a.position),
            barycentric.x * tri.a.normal + barycentric.y * tri.b.normal + barycentric.z * tri.c.normal,
            dot(tri.a.normal, tri.a.normal) != 0.0 && dot(tri.b.normal, tri.b.normal) != 0.0 && dot(tri.c.normal, tri.c.normal) != 0.0
        ));
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
    return make_inf_ray(camera.position, dir);
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
fn mean(v: vec3f) -> f32 {
    return (v.x + v.y + v.z) / 3.0;
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
    // if cos_theta(context.incident_dir) <= 0 || cos_theta(context.outgoing_dir) <= 0 {
    //     return vec3f(0.0);
    // }
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

fn conductor_eval(context: BSDFContext, material: Material) -> vec3f {
    return vec3f(0.0);
}
fn conductor_pdf(context: BSDFContext, material: Material) -> f32 {
    return 0.0;
}
fn conductor_sample(incident_dir: vec3f, rng_sample: vec2f, material: Material) -> BSDFContext {
    var context = BSDFContext();
    context.incident_dir = incident_dir;

    if cos_theta(incident_dir) <= 0.0 {
        context.color = vec3f(0.0);
        return context;
    }

    context.outgoing_dir = reflect(incident_dir);
    context.color = fresnel_conductor(context.outgoing_dir, material.eta, material.k);
    context.relative_refractive_index = 1.0;

    return context;
}

fn fresnel_dielectric(cos_theta_I: f32, extIOR: f32, intIOR: f32) -> f32 {
    if extIOR == intIOR { return 0.0; }

    let etaI = select(intIOR, extIOR, cos_theta_I >= 0.0);
    let etaT = select(extIOR, intIOR, cos_theta_I >= 0.0);
    let abs_cos_theta_I = abs(cos_theta_I);

    let eta = etaI / etaT;
    let sin_theta_t_sqr = eta * eta * (1.0 - abs_cos_theta_I * abs_cos_theta_I);

    if sin_theta_t_sqr > 1.0 { return 1.0; }

    let cos_theta_T = sqrt(1.0 - sin_theta_t_sqr);

    let Rs = (etaI * cos_theta_I - etaT * cos_theta_T)
           / (etaI * cos_theta_I + etaT * cos_theta_T);
    let Rp = (etaT * cos_theta_I - etaI * cos_theta_T)
           / (etaT * cos_theta_I + etaI * cos_theta_T);

    return (Rs * Rs + Rp * Rp) / 2.0;
}

fn fresnel_conductor(incident_dir: vec3f, eta: vec3f, k: vec3f) -> vec3f {
    let cosTheta = cos_theta(incident_dir);
    let cos2Theta = cosTheta * cosTheta;
    let sin2Theta = 1 - cos2Theta;
    let sinTheta = sqrt(sin2Theta);
    let tanTheta = sinTheta / cosTheta;

    let eta2 = eta * eta;
    let k2 = k * k;

    let a = eta2 - k2 - sin2Theta;
    let s = sqrt(a * a + 4.0 * eta2 * k2);

    let a2 = 0.5 * (s + a);
    let b2 = 0.5 * (s - a);
    let c = a2 + b2;

    let Rs = (c - 2.0 * eta * cosTheta + vec3f(cos2Theta)) / 
             (c + 2.0 * eta * cosTheta + vec3f(cos2Theta));
    let Rp = (c - 2.0 * eta * sinTheta * tanTheta + vec3f(sin2Theta * tanTheta * tanTheta)) /
             (c + 2.0 * eta * sinTheta * tanTheta + vec3f(sin2Theta * tanTheta * tanTheta));

    return 0.5 * (Rs + Rp);
}

fn dielectric_eval(context: BSDFContext, material: Material) -> vec3f {
    return vec3f(0.0);
}
fn dielectric_pdf(context: BSDFContext, material: Material) -> f32 {
    return 0.0;
}
fn dielectric_sample(incident_dir: vec3f, rng_sample: vec2f, material: Material) -> BSDFContext {
    let entering = cos_theta(incident_dir) > 0.0;

    let etaI = select(material.intIOR, material.extIOR, entering);
    let etaT = select(material.extIOR, material.intIOR, entering);
    let eta = etaI / etaT;

    let normal = select(vec3f(0.0, 0.0, -1.0), vec3f(0.0, 0.0, 1.0), entering);
    let cos_theta_I = dot(incident_dir, normal);

    let fresnel = fresnel_dielectric(cos_theta_I, etaI, etaT);

    var context = BSDFContext();
    context.incident_dir = incident_dir;

    if rng_sample.y < fresnel { // reflect
        context.outgoing_dir = reflect(incident_dir);
        context.color = vec3f(1.0);
    } else { // refract
        context.outgoing_dir = refract(incident_dir, normal, etaI / etaT);
        context.relative_refractive_index = eta;
        context.color = vec3f(fresnel + (eta * eta * (1.0 - fresnel)));
    }

    return context;
}


fn eval_bsdf_pdf(context: BSDFContext, material: Material) -> f32 {
    if material.bsdf == 0 {
        return diffuse_pdf(context, material);
    } else if material.bsdf == 1 {
        return phong_pdf(context, material);
    } else if material.bsdf == 2 {
        return dielectric_pdf(context, material);
    } else if material.bsdf == 3 {
        return conductor_pdf(context, material);
    }
    return 0.0;
}
fn eval_bsdf(context: BSDFContext, material: Material) -> vec3f {
    if material.bsdf == 0 {
        return diffuse_eval(context, material);
    } else if material.bsdf == 1 {
        return phong_eval(context, material);
    } else if material.bsdf == 2 {
        return dielectric_eval(context, material);
    } else if material.bsdf == 3 {
        return conductor_eval(context, material);
    }
    return vec3f(0.0);
}
fn sample_bsdf(incident_dir: vec3f, rng_sample: vec2f, material: Material) -> BSDFContext {
    if material.bsdf == 0 {
        return diffuse_sample(incident_dir, rng_sample, material);
    } else if material.bsdf == 1 {
        return phong_sample(incident_dir, rng_sample, material);
    } else if material.bsdf == 2 {
        return dielectric_sample(incident_dir, rng_sample, material);
    } else if material.bsdf == 3 {
        return conductor_sample(incident_dir, rng_sample, material);
    }
    return BSDFContext();
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

struct MetropolisState {
    last_sample: vec2f,
    last_eval_sample: BSDFContext,
    next_sample: vec2f,
    first: bool,
}

fn metropolis_mutate(last: vec2f, rng_state: ptr<function, RngState>) -> vec2f {
    if next_random(rng_state) < 0.1 {
        return next_random_2d(rng_state);
    }
    return abs(fract(last + (next_random_2d(rng_state) - vec2f(0.5)) / 64.0 ));
}
fn next_proposed_metropolis(state: ptr<function, MetropolisState>, rng_state: ptr<function, RngState>) -> vec2f {
    let next = metropolis_mutate((*state).last_sample, rng_state);
    (*state).next_sample = next;
    return next;
}
fn next_metropolis(eval_proposed: BSDFContext, state: ptr<function, MetropolisState>, rng_state: ptr<function, RngState>) -> BSDFContext {
    let acceptance_ratio = min(1.0, mean(eval_proposed.color) / mean((*state).last_eval_sample.color));
    if ((*state).first || next_random(rng_state) < acceptance_ratio) {
        (*state).first = false;
        (*state).last_sample = (*state).next_sample;
        (*state).last_eval_sample = eval_proposed;

        return eval_proposed;
    }
    return (*state).last_eval_sample;
}

fn metropolis_init() -> MetropolisState {
    return MetropolisState(
        vec2f(0.0),
        BSDFContext(),
        vec2f(0.0),
        true,
    );
}

fn uv_to_rng_state(uv: vec2f) -> RngState {
    return RngState(
        bitcast<u32>(fract(sin(dot(uv.xy, vec2f(12.9898, 78.2338))) * 43758.5453123)),
        bitcast<u32>(fract(sin(dot(camera.position.xz, vec2f(91.4435, 42.1895))) * 15512.5037034)),
        bitcast<u32>(fract(sin(dot(camera.position.yz, vec2f(83.9575, 67.0150))) * 32739.7101972)),
        bitcast<u32>(fract(sin(dot(uv.yx, vec2f(39.6784, 53.7602))) * 77672.1025434)),
    );
}

struct EmitterContext {
    color: vec3f,
    emitter_pdf: f32,
    emittance_dir: vec3f,
    world_emittance_dir: vec3f, // the direction from the origin to the emittance origin
    emitter_distance: f32,
    emittance_origin: vec3f,
    light: Light,
}

fn is_emitter(material: Material) -> bool {
    return dot(material.radiance, material.radiance) != 0.0;
}
fn eval_emitter_direct(emittance_dir: vec3f, material: Material) -> vec3f {
    if abs(cos_theta(emittance_dir)) > 0.0 {
        return material.radiance;
    }
    return vec3f(0.0);
}
fn emitter_pdf_direct(square_distance: f32, emittance_dir: vec3f, mesh: NEMesh) -> f32 {
    return min(square_distance / max(EPSILON, mesh.surface_area * abs(cos_theta(emittance_dir))), INF);
}
fn sample_emitter_direct(origin: vec3f, light: Light, rng_sample: vec2f) -> EmitterContext {
    let mesh = mesh_buffer[light.mesh_index];
    let face = tri_face_buffer[mesh.face_offset + light.face_index];
    let material = material_buffer[face.material_index + mesh.material_offset];
    let transform = transform_buffer[mesh.transform_index];

    // NOTE vertex access
    let tri = get_triangle(face, mesh, transform);

    let n = normalize(cross(tri.b.position - tri.a.position, tri.c.position - tri.a.position));
    let p = uniform_triangle_to_triangle(
        square_to_uniform_triangle(rng_sample),
        tri.a.position, tri.b.position, tri.c.position
    );

    let local_frame = create_frame(n);

    var emitter_context = EmitterContext();
    emitter_context.light = light;
    emitter_context.emittance_origin = p;

    // TODO find better way of calculating length and square length
    let delta = origin - p;
    emitter_context.emitter_distance = length(delta);
    emitter_context.world_emittance_dir = delta / emitter_context.emitter_distance;
    emitter_context.emittance_dir = to_local(emitter_context.world_emittance_dir, local_frame);
    emitter_context.emitter_pdf = emitter_pdf_direct(emitter_context.emitter_distance * emitter_context.emitter_distance, emitter_context.emittance_dir, mesh);
    emitter_context.color = eval_emitter_direct(emitter_context.emittance_dir, material) / emitter_context.emitter_pdf;

    return emitter_context;
}
fn sample_random_emitter_direct(origin: vec3f, rng_sample: vec2f) -> EmitterContext {
    let light = light_buffer[u32(rng_sample.x * f32(arrayLength(&light_buffer)))];

    var emitter_context = sample_emitter_direct(origin, light, rng_sample);

    // emitter_context.color *= f32(arrayLength(&light_buffer));
    return emitter_context;
}
fn mi_weight(a: f32, b: f32) -> f32 {
    return (a + b) / 2.0;
}

fn is_nan(v: vec3f) -> bool {
    let l = dot(v,v);
    return !(l < 0.0 || l >= 0.0);
}

fn eval_bsdf_path(in_ray: Ray, rng_state: ptr<function, RngState>) -> vec3f {
    var throughput = vec3f(1.0);
    var radiance = vec3f(0.0);
    var ray = in_ray;

    var weight: f32;

    for (var bounce = 0u; bounce < camera.max_bounces; bounce++) {
        let hit_info = trace_ray(ray);
        if !hit_info.hit {
            break;
        }

        let local_frame = create_frame(hit_info.normal);
        let incident_dir = to_local(-ray.direction, local_frame);

        let material = material_buffer[hit_info.material_index];

        // TODO fix metropolis start bias
        // let bsdf_context = next_metropolis(
        //     sample_bsdf(incident_dir, next_proposed_metropolis(metropolis_state, rng_state), material),
        //     metropolis_state,
        //     rng_state,
        // );

        let bsdf_context = sample_bsdf(incident_dir, next_random_2d(rng_state), material);

        if is_emitter(material) {
            radiance += eval_emitter_direct(incident_dir, material) * throughput;
        }

        var probability_to_die = max(0.01, length(bsdf_context.color));
        if bounce >= camera.min_bounces {
            if next_random(rng_state) > probability_to_die {
                break;
            }
        } else {
            probability_to_die = 1.0;
        }

        throughput *= bsdf_context.color / probability_to_die;
        ray = make_inf_ray(hit_info.intersection, to_world(bsdf_context.outgoing_dir, local_frame));
    }

    return radiance;
}

fn gradient(t: f32) -> vec3f {
    let stop0 = 0.0;
    let stop1 = 0.33;
    let stop2 = 0.66;
    let stop3 = 1.0;

    let color0 = vec3f(1.0, 0.0, 0.0);
    let color1 = vec3f(1.0, 1.0, 0.0);
    let color2 = vec3f(0.0, 1.0, 0.0);
    let color3 = vec3f(0.0, 0.0, 1.0);

    var color: vec3f;
    color = mix(color0, color1, clamp((t - stop0) / (stop1 - stop0), 0.0, 1.0));
    color = mix(color, color2, clamp((t - stop1) / (stop2 - stop1), 0.0, 1.0));
    color = mix(color, color3, clamp((t - stop2) / (stop3 - stop2), 0.0, 1.0));

    return color;
}

fn eval_mis_path(in_ray: Ray, rng_state: ptr<function, RngState>) -> vec3f {
    var throughput = vec3f(1.0);
    var radiance = vec3f(0.0);
    var ray = in_ray;

    var weight: f32;

    for (var bounce = 0u; bounce < camera.max_bounces; bounce++) {
        let hit_info = trace_ray(ray);
        let grad = gradient(hit_info.test_count / 500.0);
        if !hit_info.hit {
            // return grad;
            // break;
        } else {
            // return grad * 0.8 + vec3f(0.0, 0.4, 0.4);
            // return gradient(hit_info.distance / 30.0);
        }

        let local_frame = create_frame(hit_info.normal);
        let incident_dir = to_local(-ray.direction, local_frame);

        if true {
            // return vec3f(cos_theta(incident_dir));
        }

        let material = material_buffer[hit_info.material_index];

        if is_emitter(material) {
            radiance += throughput * eval_emitter_direct(incident_dir, material);
        }

        // direct illumination
        {
            let emitter_context = sample_random_emitter_direct(hit_info.intersection, next_random_2d(rng_state));
            let bsdf_context = BSDFContext(
                incident_dir,
                to_local(-emitter_context.world_emittance_dir, local_frame),
                1.0,
                vec2f(1.0),
                vec3f(0.0),
            );

            let bsdf_pdf = eval_bsdf_pdf(bsdf_context, material);
            let weight = mi_weight(emitter_context.emitter_pdf, bsdf_pdf);

            if true {
                // return to_local(emitter_context.world_incident_dir, local_frame);
                // return vec3f(emitter_context.color);
                // return vec3f(cos_theta(emitter_context.emittance_dir));

                // return vec3f(emitter_context.emittance_dir.z); // / 10000000000.0 
                // return vec3f(emitter_context.emitter_pdf / 1000.0);
                // return emitter_context.color;
            }

            let contribution = 
                throughput
                * emitter_context.color
                * eval_bsdf(bsdf_context, material)
                * abs(cos_theta(bsdf_context.outgoing_dir))
                * weight;

            let shadow_ray_hit_info = trace_ray(make_ray(hit_info.intersection, -emitter_context.world_emittance_dir, emitter_context.emitter_distance - 0.02));
            if !shadow_ray_hit_info.hit {
                radiance += contribution;
            }

            if true {
                // return radiance;
                // return vec3f(- shadow_ray_hit_info.distance + emitter_context.emitter_distance);
            }
        }

        // continue with bsdf
        {
            let bsdf_context = sample_bsdf(incident_dir, next_random_2d(rng_state), material);
            throughput *= bsdf_context.color;

            if bounce >= camera.min_bounces {
                let probability_to_die = max(0.01, length(bsdf_context.color));
                if next_random(rng_state) > probability_to_die {
                    break;
                }
                throughput /= probability_to_die;
            }

            ray = make_inf_ray(hit_info.intersection, to_world(bsdf_context.outgoing_dir, local_frame));
        }
    }

    return radiance;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4f {
    var rng_state = uv_to_rng_state(in.uv);
    // var metropolis_state = metropolis_init();
    var color = vec3f(0.0);
    for (var sample_index = 0u; sample_index < camera.samples; sample_index++) {
        color += eval_mis_path(ray_from_uv(in.uv), &rng_state);
        // color += eval_bsdf_path(ray_from_uv(in.uv), &rng_state);
    }
    return vec4f(color / f32(camera.samples), 1.0);
}