use bevy::prelude::{Deref, Query, Res, Resource};
use bevy::render::render_resource::StorageBuffer;
use crate::extract::GpuNEMesh;
use crate::types::NEMesh;

#[derive(Resource, Deref, Default)]
pub struct ModelBuffer(std::sync::Mutex<StorageBuffer<Vec<GpuNEMesh>>>);

pub fn fill_buffers(
    model_buffer: Res<ModelBuffer>,
    meshes: Query<(&GpuNEMesh)>
) {
    let Ok(mut model_buffer) = model_buffer.lock() else { return; };

    let mut all_meshes = Vec::new();
    for (_index, (mesh)) in meshes.iter().enumerate() {
        all_meshes.push(mesh.clone());
    }

    model_buffer.set(all_meshes);
}