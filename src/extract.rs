use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::extract_component::ExtractComponent;
use bevy::render::render_resource::ShaderType;
use crate::render::{GpuRaytracingCamera, RaytracingCamera};

#[derive(Component, ShaderType, Clone)]
pub struct GpuNEMesh {
    pub vertex_offset: u32,
    pub index_start: u32,
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

        Some(gpu_camera)
    }
}