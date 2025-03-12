#import "shaders/const.wgsl"::{PI, INV_PI}

fn square_to_cosine_hemisphere_pdf(v: vec3<f32>) -> float {
    return max(0.0, v.z) * INV_PI;
}
fn square_to_uniform_disk(v: vec2<f32>) -> vec2<f32> {
    let theta = v.x * PI * 2.0;
    let r = sqrt(v.y);

    // TODO find more efficient way to calculate this
    return vec2<f32>(r * cos(theta), r * sin(theta));
}