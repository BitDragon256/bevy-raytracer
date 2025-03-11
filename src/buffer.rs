use bevy::prelude::{Deref, Query, Res, Resource};
use bevy::render::render_resource::StorageBuffer;
use crate::extract::GpuNEMesh;
use crate::types::{NEVertex, NEIndex, NEMesh};

#[derive(Resource, Deref, Default)]
pub struct MeshBuffer(std::sync::Mutex<StorageBuffer<Vec<GpuNEMesh>>>);

#[derive(Resource, Deref, Default)]
pub struct VertexBuffer(std::sync::Mutex<StorageBuffer<Vec<NEVertex>>>);

#[derive(Resource, Deref, Default)]
pub struct IndexBuffer(std::sync::Mutex<StorageBuffer<Vec<NEIndex>>>);

pub fn fill_buffers(
    model_buffer: Res<MeshBuffer>,
    vertex_buffer: Res<VertexBuffer>,
    index_buffer: Res<IndexBuffer>,
    mut meshes: Query<&mut NEMesh>,
) {
    let mut all_meshes = Vec::new();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for (mut mesh) in &mut meshes {
        all_meshes.push(GpuNEMesh {
            vertex_offset: vertices.len() as u32,
            index_start: indices.len() as u32,
            index_count: mesh.indices.len() as u32,
        });
        vertices.append(&mut mesh.vertices);
        indices.append(&mut mesh.indices);
    }

    let Ok(mut model_buffer) = model_buffer.lock() else { return; };
    let Ok(mut vertex_buffer) = vertex_buffer.lock() else { return; };
    let Ok(mut index_buffer) = index_buffer.lock() else { return; };

    model_buffer.set(all_meshes);
    vertex_buffer.set(vertices);
    index_buffer.set(indices);
}