use bevy::prelude::{Deref, Query, Res, Resource};
use bevy::render::render_resource::StorageBuffer;
use crate::extract::GpuNEMesh;
use crate::types::{NEVertex, NEMesh, RaytracingMaterial, NETriFace};

#[derive(Resource, Deref, Default)]
pub struct MeshBuffer(std::sync::Mutex<StorageBuffer<Vec<GpuNEMesh>>>);

#[derive(Resource, Deref, Default)]
pub struct VertexBuffer(std::sync::Mutex<StorageBuffer<Vec<NEVertex>>>);

#[derive(Resource, Deref, Default)]
pub struct TriFaceBuffer(std::sync::Mutex<StorageBuffer<Vec<NETriFace>>>);

#[derive(Resource, Deref, Default)]
pub struct MaterialBuffer(std::sync::Mutex<StorageBuffer<Vec<RaytracingMaterial>>>);

pub fn fill_buffers(
    model_buffer: Res<MeshBuffer>,
    vertex_buffer: Res<VertexBuffer>,
    tri_face_buffer: Res<TriFaceBuffer>,
    material_buffer: Res<MaterialBuffer>,
    mut meshes: Query<(&mut NEMesh, &RaytracingMaterial)>,
) {
    let mut all_meshes = Vec::new();
    let mut vertices = Vec::new();
    let mut tri_faces = Vec::new();
    let mut materials = Vec::new();

    for (mut mesh, material) in &mut meshes {
        all_meshes.push(GpuNEMesh {
            vertex_offset: vertices.len() as u32,
            face_start: tri_faces.len() as u32,
            face_count: mesh.faces.len() as u32,
        });
        vertices.append(&mut mesh.vertices);
        tri_faces.append(&mut mesh.faces);
        materials.push(material.clone());
    }

    let Ok(mut model_buffer) = model_buffer.lock() else { return; };
    let Ok(mut vertex_buffer) = vertex_buffer.lock() else { return; };
    let Ok(mut tri_face_buffer) = tri_face_buffer.lock() else { return; };
    let Ok(mut material_buffer) = material_buffer.lock() else { return; };

    model_buffer.set(all_meshes);
    vertex_buffer.set(vertices);
    tri_face_buffer.set(tri_faces);
    material_buffer.set(materials);
}