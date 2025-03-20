use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::extract_component::ExtractComponent;
use bevy::render::render_resource::ShaderType;
use bvh::aabb::Aabb;
use crate::render::{GpuRaytracingCamera, RaytracingCamera};

#[derive(ShaderType)]
pub struct GpuNEMesh {
    pub vertex_offset: u32,
    pub face_offset: u32,
    pub bvh_root: u32,
    pub bvh_size: u32,
    pub material_offset: u32,
    pub transform_index: u32,
    pub surface_area: f32,
}

#[derive(ShaderType, Component, Clone, Debug)]
pub struct GpuTransform {
    pub transform: Mat4,
}

impl GpuTransform {
    pub fn new(mat: Mat4) -> Self {
        Self {
            transform: mat
        }
    }
}

impl ExtractComponent for GpuTransform {
    type QueryData = (&'static GlobalTransform);
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(
        (transform): QueryItem<'_,
        Self::QueryData>
    ) -> Option<Self::Out> {
        Some(Self::new(transform.compute_matrix()))
    }
}

#[derive(ShaderType, Debug)]
pub struct GpuBvhNode {
    pub min: Vec3,
    pub max: Vec3,
    pub entry_index: u32,
    pub exit_index: u32,
    pub shape_index: u32,
}

impl GpuBvhNode {
    pub fn from_bvh(
        aabb: &Aabb<f32, 3>,
        entry_index: u32,
        exit_index: u32,
        shape_index: u32,
    ) -> Self {
        Self {
            min: Vec3::new(aabb.min.x, aabb.min.y, aabb.min.z),
            max: Vec3::new(aabb.max.x, aabb.max.y, aabb.max.z),
            entry_index,
            exit_index,
            shape_index,
        }
    }
}

impl ExtractComponent for GpuRaytracingCamera {
    type QueryData = (
        &'static RaytracingCamera,
        &'static GlobalTransform,
        &'static Projection,
    );
    type QueryFilter = ();
    type Out = (GpuRaytracingCamera);

    fn extract_component(item: QueryItem<'_, Self::QueryData>) -> Option<Self::Out> {
        let (
            camera,
            global_transform,
            projection
        ) = item;

        let gpu_camera = match *projection {
            Projection::Orthographic(_) => return None,
            Projection::Perspective(PerspectiveProjection{
                fov,
                aspect_ratio,
                near,
                far
            }) => {
                GpuRaytracingCamera {
                    min_bounces: camera.min_bounces,
                    max_bounces: camera.max_bounces,
                    bounce_probability: camera.bounce_probability,
                    samples: camera.samples,
                    pos: global_transform.translation(),
                    dir: global_transform.forward().as_vec3(),
                    up: global_transform.up().as_vec3(),

                    aspect: aspect_ratio,
                    near,
                    far,
                    fov,
                }
            }
        };

        // println!("camera:");
        // println!("bounces: {}", camera.bounces);
        // println!("samples: {}", camera.samples);
        // println!("pos: {:?}", gpu_camera.pos);
        // println!("dir: {:?}", gpu_camera.dir);
        // println!("up: {:?}", gpu_camera.up);
        // println!("aspect: {}", gpu_camera.aspect);
        // println!("near: {}", gpu_camera.near);
        // println!("far: {}", gpu_camera.far);
        // println!("fov: {}", gpu_camera.fov);

        Some(gpu_camera)
    }
}