#import "shaders/types.wgsl"::{Ray, HitInfo}
#import "shaders/const.wgsl"::EPSILON

fn intersect_triangle(ray: Ray, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> HitInfo {
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

	return intersection;
}