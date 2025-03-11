use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::extract_component::ExtractComponent;
use bevy::render::render_resource::ShaderType;
use crate::render::{GpuRaytracingCamera, RaytracingCamera};
use crate::types::{NEIndex, NEMesh, NEVertex};

#[derive(ShaderType)]
pub struct GpuNEMesh {
    pub vertex_offset: u32,
    pub index_start: u32,
    pub index_count: u32,
}

// #[derive(Component, Clone)]
// pub struct ExtractedNEMesh {
//     pub vertices: Vec<NEVertex>,
//     pub indices: Vec<NEIndex>,
// }
//
// impl ExtractComponent for ExtractedNEMesh {
//     type QueryData = (
//         &'static NEMesh,
//         &'static GlobalTransform,
//     );
//     type QueryFilter = ();
//     type Out = Self;
//
//     fn extract_component(item: QueryItem<'_, Self::QueryData>) -> Option<Self::Out> {
//         let (mesh, global_transform) = item;
//         Some(Self {
//             vertices: mesh.vertices.iter().map(|v|
//                 NEVertex {
//                     cell: v.clone().cell,
//                     pos: v.pos + global_transform.translation()
//                 }).collect(),
//             indices: mesh.indices.clone(),
//         })
//     }
// }

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
                    bounces: camera.bounces,
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