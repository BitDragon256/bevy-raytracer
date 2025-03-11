use bevy::app::{App, Plugin};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin};
use bevy::render::render_graph::{RenderGraphApp, RenderLabel, ViewNodeRunner};
use bevy::render::render_resource::ShaderType;
use bevy::render::{Render, RenderApp, RenderSet};
use bevy::transform::helper::ComputeGlobalTransformError;
use crate::buffer::{fill_buffers, IndexBuffer, MeshBuffer, VertexBuffer};
use crate::extract::GpuNEMesh;
use crate::pipeline::RaytracingPipeline;
use crate::raytracing_render_node::{RaytraceLabel, RaytracingRenderNode};

pub fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)).looking_at(Vec3::default(), Vec3::Y),
        Camera {
            clear_color: Color::BLACK.into(),
            ..default()
        },
        RaytracingCamera {
            bounces: 1,
            samples: 1,
        }
    ));
}

#[derive(Component)]
pub struct RaytracingCamera {
    pub bounces: u32,
    pub samples: u32,
}

#[derive(Component, ShaderType, Clone, Default)]
pub struct GpuRaytracingCamera {
    pub bounces: u32,
    pub samples: u32,

    pub pos: Vec3,
    pub dir: Vec3,
    pub up: Vec3,

    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    pub fov: f32,
}

pub struct RaytracingPlugin;

impl Plugin for RaytracingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<GpuRaytracingCamera>::default(),
            UniformComponentPlugin::<GpuRaytracingCamera>::default(),

            // ExtractComponentPlugin::<GpuNEMesh>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else { return; };

        render_app
            .add_render_graph_node::<ViewNodeRunner<RaytracingRenderNode>>(
                Core3d,
                RaytraceLabel,
            )
            .add_render_graph_edges( // TODO find good place for raytracing node
                Core3d,
                (
                    Node3d::Tonemapping,
                    RaytraceLabel,
                    Node3d::EndMainPassPostProcessing,
                )
            );

        // TODO own function to initialize buffers
        render_app.init_resource::<MeshBuffer>();
        render_app.init_resource::<VertexBuffer>();
        render_app.init_resource::<IndexBuffer>();
        render_app.add_systems(Render, fill_buffers.in_set(RenderSet::PrepareResources));

        println!(">> raytracing plugin build() done");
    }
    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else { return; };

        render_app.init_resource::<RaytracingPipeline>();

        println!(">> raytracing plugin finish() done")
    }
}