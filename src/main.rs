use bevy::prelude::*;
use bevy::window::PresentMode;
use crate::render::{setup, RaytracingPlugin};

mod render;
mod types;
mod raytracing_render_node;
mod pipeline;
mod bvh;
mod extract;
mod mesh;
mod buffer;
mod scene;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                present_mode: PresentMode::Mailbox,
                ..default()
            }),
            ..default()
        }), RaytracingPlugin))
        .add_systems(Startup, setup)
        .run();
}