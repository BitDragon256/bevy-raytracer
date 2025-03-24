use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::extract_component::ExtractComponent;
use bevy::render::render_resource::ShaderType;
use bvh::aabb::Aabb;
use bvh::bvh::BvhNode;
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
    pub flattened_bvh: u32, // bool
}

#[derive(ShaderType, Component, Clone, Debug)]
pub struct GpuTransform {
    pub translate: Vec3,
    pub scale: Vec3,
    pub rotate: Mat3,
    pub inv_rotate: Mat3,
}

impl GpuTransform {
    pub fn new(translate: Vec3, scale: Vec3, rotate: Quat) -> Self {
        Self {
            translate,
            scale,
            rotate: Mat3::from_quat(rotate),
            inv_rotate: Mat3::from_quat(rotate).inverse(),
        }
    }
    pub fn apply(&self, v: Vec3) -> Vec3 {
        self.rotate.mul_vec3(v) * self.scale + self.translate
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
        Some(Self::new(transform.translation(), transform.scale(), transform.rotation()))
    }
}

#[derive(ShaderType, Debug, Clone)]
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
    pub fn empty() -> Self {
        Self {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
            entry_index: u32::MAX,
            exit_index: u32::MAX,
            shape_index: u32::MAX,
        }
    }
}

pub fn to_gpu_bvh(bvh_nodes: &Vec<BvhNode<f32, 3>>) -> Vec<GpuBvhNode> {
    let mut bvh = vec![GpuBvhNode::empty(); bvh_nodes.len()];
    for i in 0..bvh_nodes.len() {
        match bvh_nodes[i] {
            BvhNode::Leaf { shape_index: shape, .. } => bvh[i].shape_index = shape as u32,
            BvhNode::Node { parent_index: _, child_l_index, child_l_aabb, child_r_index, child_r_aabb } => {
                bvh[child_l_index].min = Vec3::new(child_l_aabb.min.x, child_l_aabb.min.y, child_l_aabb.min.z);
                bvh[child_l_index].max = Vec3::new(child_l_aabb.max.x, child_l_aabb.max.y, child_l_aabb.max.z);

                bvh[child_r_index].min = Vec3::new(child_r_aabb.min.x, child_r_aabb.min.y, child_r_aabb.min.z);
                bvh[child_r_index].max = Vec3::new(child_r_aabb.max.x, child_r_aabb.max.y, child_r_aabb.max.z);

                bvh[i].entry_index = child_l_index as u32;
                bvh[i].exit_index = child_r_index as u32;
            }
        }
    }
    bvh
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