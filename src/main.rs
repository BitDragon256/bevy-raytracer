use bevy::prelude::*;
use crate::render::{setup, RaytracingPlugin};

mod render;
mod types;
mod raytracing_render_node;
mod pipeline;
mod bvh;
mod extract;
mod mesh;
mod buffer;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, RaytracingPlugin))
        .add_systems(Startup, setup)
        .run();
}