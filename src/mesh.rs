// TODO
// custom render graph node that gets a vertex stage where all models in the current scene
// are displayed and then saved into a storage buffer to be accessed by the raytracer
// here we probably also can compute the BVH tree

use std::rc::Rc;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::{BHShape, BoundingHierarchy};
use bvh::bvh::Bvh;
use nalgebra::{Point3};
use crate::types::{NEMesh, NETriFace, NEVertex};

struct NETriangle<'a> {
    pub face: NETriFace,
    vertices: &'a Vec<NEVertex>,
    pub bvh_index: usize,
}

impl Bounded<f32, 3> for NETriangle<'_> {
    fn aabb(&self) -> Aabb<f32, 3> {
        let a = self.vertices[self.face.a as usize].pos.clone();
        let b = self.vertices[self.face.b as usize].pos.clone();
        let c = self.vertices[self.face.c as usize].pos.clone();

        Aabb::with_bounds(
            Point3::new(a.x.min(b.x).min(c.x), a.y.min(b.y).min(c.y), a.z.min(b.z).min(c.z)),
            Point3::new(a.x.max(b.x).max(c.x), a.y.max(b.y).max(c.y), a.z.max(b.z).max(c.z)),
        )
    }
}
impl BHShape<f32, 3> for NETriangle<'_> {
    fn set_bh_node_index(&mut self, index: usize) {
        self.bvh_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.bvh_index
    }
}

impl NEMesh {
    pub(crate) fn new(vertices: Vec<NEVertex>, faces: Vec<NETriFace>) -> Self {
        let (bvh, bvh_faces) = {
            let mut bvh_faces: Vec<NETriangle> = faces.into_iter().map(|face| NETriangle {
                face,
                vertices: &vertices,
                bvh_index: usize::MAX,
            }).collect();
            (
                Bvh::build_par(&mut bvh_faces),
                bvh_faces.iter().map(|face| face.face.with_bvh_index(face.bvh_index as u32)).collect(),
            )
        };
        NEMesh {
            vertices,
            faces: bvh_faces,
            bvh,
        }
    }
}
