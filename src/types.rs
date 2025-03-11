use bevy::math::Vec3;
use bevy::prelude::Mesh;
use bevy::render::render_resource::ShaderType;

pub struct Cell {
    sockets: Vec<Socket>,
}

#[derive(ShaderType)]
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

#[derive(ShaderType)]
pub struct NEVertex {
    pos: Vec3,
    cell: CellRef,
}

pub struct NEMesh {
    vertices: Vec<NEVertex>,
    indices: Vec<u32>,
}