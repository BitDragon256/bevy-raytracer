use bevy::math::Vec3;
use bevy::prelude::{Component, Mesh};
use bevy::render::extract_component::ExtractComponent;
use bevy::render::render_resource::ShaderType;
use bvh::bvh::Bvh;

pub struct Cell {
    sockets: Vec<Socket>,
    bvh: Bvh<f32, 3>,
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

#[derive(ShaderType, Clone, Debug)]
pub struct NETriFace {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    material_index: u32,
    bvh_index: u32,
}

impl NETriFace {
    pub fn new(a: u32, b: u32, c: u32, mat: u32) -> Self {
        NETriFace {
            a, b, c,
            material_index: mat,
            bvh_index: u32::MAX,
        }
    }
    pub fn with_bvh_index(&self, index: u32) -> Self {
        let mut face = self.clone();
        face.bvh_index = index;
        face
    }
}

#[derive(Component, ExtractComponent, Clone)]
pub struct NEMesh {
    pub vertices: Vec<NEVertex>,
    pub faces: Vec<NETriFace>,
    pub bvh: Bvh<f32, 3>,
}

// BSDF overview:
// 0 -> diffuse
// 1 -> phong
// 2 -> ...

#[derive(ShaderType, Component, ExtractComponent, Clone)]
pub struct RaytracingMaterial {
    pub bsdf: u32,
    pub radiance: Vec3,
    pub albedo: Vec3,
    pub specular: Vec3,
    pub exponent: f32,
}

pub fn string_to_bsdf(name: &str) -> u32 {
    match name {
        "diffuse" => 0,
        "phong" => 1,
        _ => 0
    }
}