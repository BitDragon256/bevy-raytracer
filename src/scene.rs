use std::fs::File;
use std::io::BufReader;
use bevy::prelude::*;
use bevy::prelude::Projection::Perspective;
use bevy_flycam::{FlyCam, MovementSettings};
use obj::{load_obj, Obj, ObjResult};
use serde_json::{Number, Value};
use crate::render::RaytracingCamera;
use crate::types::{string_to_bsdf, CellRef, NEMesh, NETriFace, NEVertex, RaytracingMaterial};

fn array_to_vec(array: &Value) -> Vec3 {
    match array {
        Value::Array(v) => Vec3::from_slice(v.iter().map(|n| match n {
            Value::Number(u) => u.as_f64().unwrap() as f32,
            _ => panic!("{:?} is no number", n)
        }).collect::<Vec<_>>().as_slice()),
        _ => panic!("{:?} is no vector", array)
    }
}
fn cond_array_to_vec(array: Option<&Value>, default: Vec3) -> Vec3 {
    match array {
        None => default,
        Some(v) => array_to_vec(v)
    }
}
fn cond_f32(v: Option<&Value>, default: f32) -> f32 {
    match v {
        Some(Value::Number(n)) => n.as_f64().unwrap() as f32,
        _ => default
    }
}
fn cond_u32(v: Option<&Value>, default: u32) -> u32 {
    match v {
        Some(Value::Number(n)) => n.as_u64().unwrap() as u32,
        _ => default
    }
}
fn cond_bool(v: Option<&Value>, default: bool) -> bool {
    match v {
        Some(Value::Bool(b)) => *b,
        _ => default
    }
}
fn cond_string(v: Option<&Value>, default: &str) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        _ => default.to_string()
    }
}

fn new_object_reader(filename: &str) -> BufReader<File> {
    BufReader::new(File::open(format!("assets/scenes/{}", filename)).unwrap_or_else(|_| panic!("object file {} does not exist", filename)))
}

fn vec_to_tri_faces(indices: &Vec<u16>) -> Vec<NETriFace> {
    indices.chunks_exact(3).map(|arr| {
        NETriFace::new(
            arr[0] as u32,
            arr[1] as u32,
            arr[2] as u32,
            0
        )
    }).collect()
}

fn load_object(filename: &str) -> NEMesh {
    let maybe_model: ObjResult<Obj<obj::Vertex>> = load_obj(new_object_reader(filename));
    if let Ok(model) = maybe_model {
        return NEMesh::new(
            model.vertices.iter().map(|vertex| {
                NEVertex::with_normal (
                    Vec3::from_array(vertex.position),
                    Vec3::from_array(vertex.normal),
                    CellRef::new(0),
                )
            }).collect(),
            vec_to_tri_faces(&model.indices)
        );
    }

    let maybe_model: ObjResult<Obj<obj::Position>> = load_obj(new_object_reader(filename));
    if let Ok(model) = maybe_model {
        return NEMesh::new(
            model.vertices.iter().map(|vertex| {
                NEVertex::with_pos (
                    Vec3::from_array(vertex.position),
                    CellRef::new(0),
                )
            }).collect(),
            vec_to_tri_faces(&model.indices)
        );
    }

    panic!("could not read object in file {filename}");
}

pub fn load_scene(
    mut commands: Commands,
    mut movement_settings: ResMut<MovementSettings>,
    filename: &str,
) {
    let scene_folder = filename.chars().take(filename.rfind("/").unwrap_or(filename.len() - 1)).collect::<String>();
    let scene_reader = BufReader::new(File::open(format!("assets/scenes/{}", filename)).expect("given file does not exist"));
    let json_scene: Value = serde_json::from_reader(scene_reader).expect("scene json was not well-formatted");

    match json_scene.get("camera") {
        Some(Value::Object(camera)) => {
            let pos = cond_array_to_vec(camera.get("origin"), Vec3::ZERO);
            let target = cond_array_to_vec(camera.get("target"), pos + Vec3::Z);
            let up = cond_array_to_vec(camera.get("up"), Vec3::Y);
            let fov = cond_f32(camera.get("fov"), 60f32);
            let sample_count = cond_u32(camera.get("sample_count"), 64);
            let sampler = cond_string(camera.get("sampler"), "independent");
            let max_bounces = cond_u32(camera.get("max_bounces"), 10);
            let min_bounces = cond_u32(camera.get("min_bounces"), 3);
            let speed = cond_f32(camera.get("speed"), 180f32);

            movement_settings.speed = speed;

            commands.spawn((
                Camera3d::default(),
                Perspective(PerspectiveProjection{
                    fov,
                    ..default()
                }),
                Transform::from_translation(pos).looking_at(target, up),
                Camera {
                    clear_color: Color::BLACK.into(),
                    ..default()
                },
                RaytracingCamera {
                    min_bounces,
                    max_bounces,
                    bounce_probability: 0.9,
                    samples: sample_count,
                },
                FlyCam
            ));
        }
        _ => panic!("no camera found"),
    };
    match json_scene.get("meshes") {
        Some(Value::Array(meshes)) => {
            for mesh in meshes {
                match mesh {
                    Value::Object(mesh_json) => {
                        if mesh_json.contains_key("invisible") {
                            continue;
                        }

                        let Value::String(object) = mesh_json.get("object").expect("mesh should have attached obj file") else { panic!("object value should be string") };
                        let mut model = load_object(format!("{}/{}", scene_folder, object).as_str());
                        model.flattened_bvh = cond_bool(mesh_json.get("flattened bvh"), true);

                        let (translate, scale, rotate) = match mesh_json.get("transform") {
                            Some(Value::Object(transform)) => (
                                cond_array_to_vec(transform.get("translate"), Vec3::ZERO),
                                cond_array_to_vec(transform.get("scale"), Vec3::ONE),
                                cond_array_to_vec(transform.get("rotate"), Vec3::ZERO),
                            ),
                            _ => (Vec3::ZERO, Vec3::ONE, Vec3::ZERO)
                        };

                        let bsdf = mesh_json.get("bsdf").expect("mesh should have attached bsdf");
                        let bsdf_type: String = bsdf.get("type").expect("bsdf should have type").to_string();
                        let albedo = cond_array_to_vec(bsdf.get("albedo"), Vec3::ONE);
                        let specular = cond_array_to_vec(bsdf.get("specular"), Vec3::ONE);
                        let radiance = cond_array_to_vec(bsdf.get("radiance"), Vec3::ZERO);
                        let exponent = cond_f32(bsdf.get("exponent"), 0.0f32);
                        let ext_ior = cond_f32(bsdf.get("extIOR"), 1f32);
                        let int_ior = cond_f32(bsdf.get("intIOR"), 1f32);
                        let eta = cond_array_to_vec(bsdf.get("eta"), Vec3::ONE);
                        let k = cond_array_to_vec(bsdf.get("k"), Vec3::ONE);

                        commands.spawn((
                            Transform::from_translation(translate).with_scale(scale).with_rotation(Quat::from_euler(EulerRot::XYZ, rotate.x.to_radians(), rotate.y.to_radians(), rotate.z.to_radians())),
                            model,
                            RaytracingMaterial {
                                bsdf: string_to_bsdf(&bsdf_type),
                                albedo, specular, exponent, radiance,
                                ext_ior, int_ior,
                                eta, k,
                            }
                        ));
                    }
                    _ => panic!("mesh {:?} is in wrong format", mesh)
                }
            }
        }
        _ => println!(">> note: no meshes in scene")
    }
}