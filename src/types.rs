use bevy::math::Vec3;
use bevy::prelude::{Component, Mesh};
use bevy::render::extract_component::ExtractComponent;
use bevy::render::render_resource::ShaderType;

pub struct Cell {
    sockets: Vec<Socket>,
}

#[derive(ShaderType, Clone)]
pub struct CellRef {
    index: u32,
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

#[derive(ShaderType, Clone)]
pub struct NEVertex {
    pub pos: Vec3,
    pub cell: CellRef,
}

pub type NEIndex = u32;

#[derive(Component, ExtractComponent, Clone)]
pub struct NEMesh {
    pub vertices: Vec<NEVertex>,
    pub indices: Vec<u32>,
}