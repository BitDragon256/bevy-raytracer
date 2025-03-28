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
    pub normal: Vec3,
    pub texture: Vec3,
    pub cell: CellRef,
}

impl NEVertex {
    pub fn with_pos(pos: Vec3, cell: CellRef) -> Self {
        Self::with_normal(pos, Vec3::ZERO, cell)
    }
    pub fn with_normal(pos: Vec3, normal: Vec3, cell: CellRef) -> Self {
        Self::new(pos, normal, Vec3::ZERO, cell)
    }
    pub fn new(pos: Vec3, normal: Vec3, texture: Vec3, cell: CellRef) -> Self {
        NEVertex {
            pos, normal, texture, cell
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
    pub flattened_bvh: bool,
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
    // dielectric
    pub ext_ior: f32,
    pub int_ior: f32,
    // conductor
    pub eta: Vec3,
    pub k: Vec3,
}

#[derive(ShaderType)]
pub struct RaytracingLight {
    pub face_index: u32,
    pub mesh_index: u32,
}

impl RaytracingLight {
    pub fn new(face_index: u32, mesh_index: u32) -> Self {
        Self {
            face_index,
            mesh_index,
        }
    }
}

pub fn string_to_bsdf(name: &str) -> u32 {
    match name.strip_suffix("\"").unwrap().strip_prefix("\"").unwrap() {
        "diffuse" => 0,
        "phong" => 1,
        "dielectric" => 2,
        "conductor" => 3,
        _ => 0
    }
}