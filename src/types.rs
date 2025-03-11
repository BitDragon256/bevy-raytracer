use bevy::math::Vec3;
use bevy::prelude::{Component, Mesh};
use bevy::render::extract_component::ExtractComponent;
use bevy::render::render_resource::ShaderType;

pub struct Cell {
    sockets: Vec<Socket>,
}

#[derive(ShaderType, Clone, Debug)]
pub struct CellRef {
    index: u32,
}

impl CellRef {
    pub fn new(index: u32) -> Self {
        Self { index }
    }
}

pub struct Socket {
    pos: Vec3,
    face: Mesh,
    parent: CellRef,
    other: SocketRef,
}
pub struct SocketRef {
    index: u32,
}

#[derive(ShaderType, Clone, Debug)]
pub struct NEVertex {
    pub pos: Vec3,
    pub cell: CellRef,
}

impl NEVertex {
    pub fn new(x: f32, y: f32, z: f32, cell: CellRef) -> Self {
        NEVertex {
            pos: Vec3::new(x, y, z),
            cell,
        }
    }
}

pub type NEIndex = u32;

#[derive(Component, ExtractComponent, Clone)]
pub struct NEMesh {
    pub vertices: Vec<NEVertex>,
    pub indices: Vec<u32>,
}