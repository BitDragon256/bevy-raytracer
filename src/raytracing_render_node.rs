use bevy::core_pipeline::prepass::ViewPrepassTextures;
use bevy::ecs::query::QueryItem;
use bevy::prelude::{FromWorld, World};
use bevy::render::extract_component::{ComponentUniforms, DynamicUniformIndex};
use bevy::render::render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode};
use bevy::render::render_resource::{AsBindGroupShaderType, BindGroupEntries, Operations, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor};
use bevy::render::renderer::{RenderContext, RenderQueue};
use bevy::render::view::ViewTarget;
use crate::buffer::{TriFaceBuffer, MaterialBuffer, MeshBuffer, VertexBuffer, LowLevelAccelerationStructureBuffer, TransformBuffer};
use crate::pipeline::RaytracingPipeline;
use crate::render::GpuRaytracingCamera;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct RaytraceLabel;

pub struct RaytracingRenderNode;

impl ViewNode for RaytracingRenderNode {
    type ViewQuery = (
        &'static ViewTarget,
        // The prepass textures (depth used for blending between raster and raytraced)
        // &'static ViewPrepassTextures,
        // This makes sure the node only runs on cameras with the PostProcessSettings component
        &'static GpuRaytracingCamera,
        // As there could be multiple post-processing components sent to the GPU (one per camera),
        // we need to get the index of the one that is associated with the current view.
        &'static DynamicUniformIndex<GpuRaytracingCamera>,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (
            view_target,
            // _view_prepass_textures,
            _gpu_raytracing_camera,
            camera_index,
        ): QueryItem<'w, Self::ViewQuery>,
        world: &'w World
    ) -> Result<(), NodeRunError> {
        let raytracing_pipeline = world.resource::<RaytracingPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pipeline) = pipeline_cache.get_render_pipeline(raytracing_pipeline.pipeline_id) else { return Ok(()) };

        let camera_uniform = world.resource::<ComponentUniforms<GpuRaytracingCamera>>();
        let Some(camera_binding) = camera_uniform.uniforms().binding() else { return Ok(()) };

        let post_process = view_target.post_process_write();

        // TODO buffer lock function
        let meshes = world.resource::<MeshBuffer>();
        let mut mesh_buffer = meshes
            .lock()
            .expect("mesh buffer mutex is unexpectedly locked");

        let bvhs = world.resource::<LowLevelAccelerationStructureBuffer>();
        let mut llas_buffer = bvhs
            .lock()
            .expect("low level acceleration structure mutex is unexpectedly locked");

        let vertices = world.resource::<VertexBuffer>();
        let mut vertex_buffer = vertices
            .lock()
            .expect("vertex buffer mutex is unexpectedly locked");

        let tri_faces = world.resource::<TriFaceBuffer>();
        let mut tri_face_buffer = tri_faces
            .lock()
            .expect("index buffer mutex is unexpectedly locked");

        let materials = world.resource::<MaterialBuffer>();
        let mut material_buffer = materials
            .lock()
            .expect("material buffer mutex is unexpectedly locked");

        let transforms = world.resource::<TransformBuffer>();
        let mut transform_buffer = transforms
            .lock()
            .expect("transform buffer mutex is unexpectedly locked");

        let render_device = render_context.render_device();
        let render_queue = world.resource::<RenderQueue>();

        mesh_buffer.write_buffer(render_device, render_queue);
        llas_buffer.write_buffer(render_device, render_queue);
        vertex_buffer.write_buffer(render_device, render_queue);
        tri_face_buffer.write_buffer(render_device, render_queue);
        material_buffer.write_buffer(render_device, render_queue);
        transform_buffer.write_buffer(render_device, render_queue);

        let Some(mesh_buffer_binding) = mesh_buffer.binding() else { return Ok(()) };
        let Some(llas_buffer_binding) = llas_buffer.binding() else { return Ok(()) };
        let Some(vertex_buffer_binding) = vertex_buffer.binding() else { return Ok(()) };
        let Some(tri_face_buffer) = tri_face_buffer.binding() else { return Ok(()) };
        let Some(material_buffer_binding) = material_buffer.binding() else { return Ok(()) };
        let Some(transform_buffer_binding) = transform_buffer.binding() else { return Ok(()) };

        let bind_group = render_device.create_bind_group(
            "raytrace_bind_group",
            &raytracing_pipeline.layout,
            // matches BindGroupLayout defined in pipeline
            &BindGroupEntries::sequential((
                post_process.source,
                &raytracing_pipeline.sampler,
                camera_binding.clone(),
            )),
        );

        let buffer_bind_group = render_device.create_bind_group(
            "raytrace_geometry_bind_group",
            &raytracing_pipeline.buffer_layout,
            &BindGroupEntries::sequential((
                mesh_buffer_binding,
                llas_buffer_binding,
                vertex_buffer_binding,
                tri_face_buffer,
                material_buffer_binding,
                transform_buffer_binding,
            )),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("raytrace_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);

        render_pass.set_bind_group(0, &bind_group, &[camera_index.index()]);
        render_pass.set_bind_group(1, &buffer_bind_group, &[]);

        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

impl FromWorld for RaytracingRenderNode {
    fn from_world(world: &mut World) -> Self {
        Self {}
    }
}