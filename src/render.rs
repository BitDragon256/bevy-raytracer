use bevy::app::{App, Plugin};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::{
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin},
        render_graph::{RenderGraphApp, ViewNodeRunner},
        render_resource::ShaderType,
        {Render, RenderApp, RenderSet},
        view::screenshot::{save_to_disk, Capturing, Screenshot},
    },
    window::SystemCursorIcon,
    winit::cursor::CursorIcon,
};

use bevy_flycam::{MovementSettings, NoCameraPlayerPlugin};
use bevy_fps_counter::{FpsCounterPlugin};

use crate::buffer::{fill_buffers, TriFaceBuffer, MaterialBuffer, MeshBuffer, VertexBuffer, LowLevelAccelerationStructureBuffer, TransformBuffer, BufferCache, LightBuffer};
use crate::extract::GpuTransform;
use crate::pipeline::RaytracingPipeline;
use crate::raytracing_render_node::{RaytraceLabel, RaytracingRenderNode};
use crate::scene::load_scene;
use crate::types::{CellRef, NEMesh, NETriFace, NEVertex, RaytracingMaterial};

pub fn setup(
    commands: Commands,
    movement_settings: ResMut<MovementSettings>,
) {
    load_scene(commands, movement_settings, "sponza/sponza.json");
    // load_scene(commands, movement_settings, "cbox/cbox.json");
    // load_scene(commands, "dragon/dragon.json");
    println!(">> example scene set up");
}

#[derive(Component)]
pub struct RaytracingCamera {
    pub min_bounces: u32,
    pub max_bounces: u32,
    pub bounce_probability: f32,

    pub samples: u32,
}

#[derive(Component, ShaderType, Clone, Default)]
pub struct GpuRaytracingCamera {
    pub min_bounces: u32,
    pub max_bounces: u32,
    pub bounce_probability: f32,

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
        // TODO move this
        app
            .add_plugins((
                NoCameraPlayerPlugin,
                FpsCounterPlugin,
            ))
            .insert_resource(MovementSettings {
                sensitivity: 0.00010,
                speed: 180.0,
            })
            .add_systems(Update, (take_screenshot, screenshot_saving));

        app.add_plugins((
            ExtractComponentPlugin::<GpuRaytracingCamera>::default(),
            UniformComponentPlugin::<GpuRaytracingCamera>::default(),

            ExtractComponentPlugin::<NEMesh>::default(),
            ExtractComponentPlugin::<RaytracingMaterial>::default(),
            ExtractComponentPlugin::<GpuTransform>::default(),
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
        render_app.init_resource::<LowLevelAccelerationStructureBuffer>();
        render_app.init_resource::<VertexBuffer>();
        render_app.init_resource::<TriFaceBuffer>();
        render_app.init_resource::<MaterialBuffer>();
        render_app.init_resource::<TransformBuffer>();
        render_app.init_resource::<LightBuffer>();

        render_app.init_resource::<BufferCache>();

        render_app.add_systems(Render, fill_buffers.in_set(RenderSet::PrepareResources));

        println!(">> raytracing plugin build() done");
    }
    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else { return; };

        render_app.init_resource::<RaytracingPipeline>();

        println!(">> raytracing plugin finish() done")
    }
}

fn take_screenshot(
    mut commands: Commands,
    input: Res<ButtonInput<KeyCode>>,
    mut counter: Local<u32>,
) {
    if input.just_pressed(KeyCode::KeyR) {
        let path = format!("./screenshots/screenshot-{}.png", *counter);
        println!(">> screenshot saved under {}", path);
        *counter += 1;
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }
}

fn screenshot_saving(
    mut commands: Commands,
    screenshot_saving: Query<Entity, With<Capturing>>,
    windows: Query<Entity, With<Window>>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    match screenshot_saving.iter().count() {
        0 => {
            commands.entity(window).remove::<CursorIcon>();
        }
        x if x > 0 => {
            commands
                .entity(window)
                .insert(CursorIcon::from(SystemCursorIcon::Progress));
        }
        _ => {}
    }
}