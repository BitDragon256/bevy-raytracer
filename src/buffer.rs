use bevy::prelude::{Deref, Entity, EntityRef, GlobalTransform, Query, Res, ResMut, Resource, Transform, Vec3, World};
use bevy::render::render_resource::StorageBuffer;
use crate::extract::{GpuBvhNode, GpuNEMesh, GpuTransform};
use crate::types::{NEVertex, NEMesh, RaytracingMaterial, NETriFace, RaytracingLight};

#[derive(Resource, Deref, Default)]
pub struct MeshBuffer(std::sync::Mutex<StorageBuffer<Vec<GpuNEMesh>>>);

// contains scene and references to meshes
#[derive(Resource, Deref, Default)]
pub struct HighLevelAccelerationStructureBuffer(std::sync::Mutex<StorageBuffer<Vec<GpuBvhNode>>>);

// contains meshes and references to faces
#[derive(Resource, Deref, Default)]
pub struct LowLevelAccelerationStructureBuffer(std::sync::Mutex<StorageBuffer<Vec<GpuBvhNode>>>);

#[derive(Resource, Deref, Default)]
pub struct VertexBuffer(std::sync::Mutex<StorageBuffer<Vec<NEVertex>>>);

#[derive(Resource, Deref, Default)]
pub struct TriFaceBuffer(std::sync::Mutex<StorageBuffer<Vec<NETriFace>>>);

#[derive(Resource, Deref, Default)]
pub struct MaterialBuffer(std::sync::Mutex<StorageBuffer<Vec<RaytracingMaterial>>>);

#[derive(Resource, Deref, Default)]
pub struct TransformBuffer(std::sync::Mutex<StorageBuffer<Vec<GpuTransform>>>);

#[derive(Resource, Deref, Default)]
pub struct LightBuffer(std::sync::Mutex<StorageBuffer<Vec<RaytracingLight>>>);

#[derive(Resource)]
pub struct BufferCache {
    pub pushed: bool,
    pub iters: i32,
}

impl Default for BufferCache {
    fn default() -> Self {
        Self {
            pushed: false,
            iters: 10,
        }
    }
}

pub fn fill_buffers(
    model_buffer: Res<MeshBuffer>,
    // hlas_buffer: Res<HighLevelAccelerationStructureBuffer>,
    llas_buffer: Res<LowLevelAccelerationStructureBuffer>,
    vertex_buffer: Res<VertexBuffer>,
    tri_face_buffer: Res<TriFaceBuffer>,
    material_buffer: Res<MaterialBuffer>,
    transform_buffer: Res<TransformBuffer>,
    light_buffer: Res<LightBuffer>,
    mut buffer_cache: ResMut<BufferCache>,
    mut meshes: Query<(&mut NEMesh, &RaytracingMaterial, &GpuTransform)>,
) {
    if buffer_cache.iters < 0 {
        return;
    } else {
        // buffer_cache.pushed = true;
        println!("iterations remaining: {}", buffer_cache.iters);
        buffer_cache.iters -= 1;
    }

    let mut all_meshes = Vec::new();
    let mut gpu_bvh = Vec::new();
    let mut vertices = Vec::new();
    let mut tri_faces = Vec::new();
    let mut materials = Vec::new();
    let mut transforms = Vec::new();
    let mut lights = Vec::new();

    for (mut mesh, material, transform) in &mut meshes {
        let mut flattened_bvh = mesh.bvh.flatten_custom(&GpuBvhNode::from_bvh);

        if material.radiance.length() > 0f32 {
            let mut mesh_lights = (0..mesh.faces.len()).map(|i| RaytracingLight::new(i as u32, all_meshes.len() as u32)).collect::<Vec<_>>();
            lights.append(&mut mesh_lights);
        }

        all_meshes.push(GpuNEMesh {
            vertex_offset: vertices.len() as u32,
            face_offset: tri_faces.len() as u32,
            bvh_root: gpu_bvh.len() as u32,
            bvh_size: flattened_bvh.len() as u32,
            material_offset: materials.len() as u32,
            transform_index: transforms.len() as u32,
            surface_area: mesh.faces.iter().map(|f| {
                let a = mesh.vertices[f.a as usize].pos;
                let b = mesh.vertices[f.b as usize].pos;
                let c = mesh.vertices[f.c as usize].pos;
                return (b - a).cross(c - a).length() * 0.5;
            }).sum()
        });

        gpu_bvh.append(&mut flattened_bvh);
        vertices.append(&mut mesh.vertices);
        tri_faces.append(&mut mesh.faces);
        materials.push(material.clone());
        transforms.push(transform.clone());
    }

    if buffer_cache.iters == 0 {
        println!("{} meshes", all_meshes.len());
        println!("{} aabbs", gpu_bvh.len());
        println!("{} vertices", vertices.len());
        println!("{} tri faces", tri_faces.len());
        println!("{} materials", materials.len());
        println!("{} transforms", transforms.len());
        println!("{} lights", lights.len());
    }

    let Ok(mut model_buffer) = model_buffer.lock() else { return; };
    let Ok(mut llas_buffer) = llas_buffer.lock() else { return; };
    let Ok(mut vertex_buffer) = vertex_buffer.lock() else { return; };
    let Ok(mut tri_face_buffer) = tri_face_buffer.lock() else { return; };
    let Ok(mut material_buffer) = material_buffer.lock() else { return; };
    let Ok(mut transform_buffer) = transform_buffer.lock() else { return; };
    let Ok(mut light_buffer) = light_buffer.lock() else { return; };

    model_buffer.set(all_meshes);
    llas_buffer.set(gpu_bvh);
    vertex_buffer.set(vertices);
    tri_face_buffer.set(tri_faces);
    material_buffer.set(materials);
    transform_buffer.set(transforms);
    light_buffer.set(lights);
}