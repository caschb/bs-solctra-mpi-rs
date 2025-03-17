use crate::{
    constants::{I, MAJOR_RADIUS, MINOR_RADIUS, MIU, PI},
    point::{Point, read_from_file, write_points_to_file},
};
use clap::error::Result;
use log::debug;
use rayon::prelude::*;
use std::{error::Error, fs, io, path::Path, usize};

pub fn compute_magnetic_field(
    particle: &Point,
    coils: &Vec<Vec<Point>>,
    displacements: &Vec<Vec<Point>>,
    e_roof: &Vec<Vec<Point>>,
) -> Point {
    let multiplier = (MIU * I) / (4.0 * PI);
    let mut b = Point {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    for coil_idx in 0..coils.len() {
        for point_idx in 0..coils[coil_idx].len() - 1 {
            let rmi_a = particle.get_displacement(&coils[coil_idx][point_idx]);
            let rmf_a = particle.get_displacement(&coils[coil_idx][point_idx + 1]);
            let u = Point {
                x: multiplier * e_roof[coil_idx][point_idx].x,
                y: multiplier * e_roof[coil_idx][point_idx].y,
                z: multiplier * e_roof[coil_idx][point_idx].z,
            };
            let displacement_norm = displacements[coil_idx][point_idx].get_norm();
            let rmi_a_norm = rmi_a.get_norm();
            let rmf_a_norm = rmf_a.get_norm();
            let c = ((2.0 * displacement_norm * (rmi_a_norm + rmf_a_norm))
                / (rmi_a_norm * rmf_a_norm))
                * ((1.0)
                    / ((rmi_a_norm + rmf_a_norm) * (rmi_a_norm + rmf_a_norm)
                        - displacement_norm * displacement_norm));
            let v = Point {
                x: rmi_a.x * c,
                y: rmi_a.y * c,
                z: rmi_a.z * c,
            };
            b.x = b.x + ((u.y * v.z) - (u.z * v.y));
            b.y = b.y - ((u.x * v.z) - (u.z * v.x));
            b.z = b.z + ((u.x * v.y) - (u.y * v.x));
        }
    }
    b
}

pub fn simulate_step(
    particle: &Point,
    coils: &Vec<Vec<Point>>,
    displacements: &Vec<Vec<Point>>,
    e_roof: &Vec<Vec<Point>>,
    step_size: f64,
) -> Point {
    let mut k1 = compute_magnetic_field(particle, coils, displacements, e_roof);
    let k1norm = k1.get_norm();
    k1.x = (k1.x / k1norm) * step_size;
    k1.y = (k1.y / k1norm) * step_size;
    k1.z = (k1.z / k1norm) * step_size;
    let p1 = Point {
        x: k1.x / 2.0 + particle.x,
        y: k1.y / 2.0 + particle.y,
        z: k1.z / 2.0 + particle.z,
    };

    let mut k2 = compute_magnetic_field(&p1, coils, displacements, e_roof);
    let k2norm = k2.get_norm();
    k2.x = (k2.x / k2norm) * step_size;
    k2.y = (k2.y / k2norm) * step_size;
    k2.z = (k2.z / k2norm) * step_size;
    let p2 = Point {
        x: k2.x / 2.0 + particle.x,
        y: k2.y / 2.0 + particle.y,
        z: k2.z / 2.0 + particle.z,
    };

    let mut k3 = compute_magnetic_field(&p2, coils, displacements, e_roof);
    let k3norm = k3.get_norm();
    k3.x = (k3.x / k3norm) * step_size;
    k3.y = (k3.y / k3norm) * step_size;
    k3.z = (k3.z / k3norm) * step_size;
    let p3 = Point {
        x: k3.x + particle.x,
        y: k3.y + particle.y,
        z: k3.z + particle.z,
    };
    let mut k4 = compute_magnetic_field(&p3, coils, displacements, e_roof);
    let k4norm = k4.get_norm();
    k4.x = (k4.x / k4norm) * step_size;
    k4.y = (k4.y / k4norm) * step_size;
    k4.z = (k4.z / k4norm) * step_size;
    let mut result = Point {
        x: particle.x + (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x) / 6.0,
        y: particle.y + (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y) / 6.0,
        z: particle.z + (k1.z + 2.0 * k2.z + 2.0 * k3.z + k4.z) / 6.0,
    };

    let p = Point {
        x: result.x,
        y: result.y,
        z: 0.0,
    };
    let origin = Point {
        x: MAJOR_RADIUS * p.x / p.get_norm(),
        y: MAJOR_RADIUS * p.y / p.get_norm(),
        z: 0.0,
    };

    let distance = result.get_distance(&origin);
    if distance > MINOR_RADIUS {
        result.x = MINOR_RADIUS;
        result.y = MINOR_RADIUS;
        result.z = MINOR_RADIUS;
    }

    result
}

pub fn simulate_particles(
    particles: &mut [Point],
    total_steps: u32,
    step_size: f64,
    coils: &Vec<Vec<Point>>,
    displacements: &Vec<Vec<Point>>,
    e_roof: &Vec<Vec<Point>>,
    output_dir: &Path,
    write_frequency: u32,
    rank: i32,
) {
    let length = particles.len();
    let divergent_particle = Point {
        x: MINOR_RADIUS,
        y: MINOR_RADIUS,
        z: MINOR_RADIUS,
    };

    debug!("Total particles: {}", length);

    match write_points_to_file(&particles, output_dir, 0, rank) {
        Ok(_) => debug!("Wrote points to {:?}", output_dir),
        Err(error) => panic!("Error writing points to file. {}", error),
    };
    for step in 1..total_steps + 1 {
        particles.par_iter_mut().for_each(|particle| {
            if *particle != divergent_particle {
                *particle = simulate_step(particle, coils, displacements, e_roof, step_size);
            }
        });
        if step % write_frequency == 0 {
            match write_points_to_file(&particles, output_dir, step, rank) {
                Ok(_) => debug!("Wrote points to {:?}", output_dir),
                Err(error) => panic!("Error writing points to file. {}", error),
            };
        }
    }
}

pub fn read_coil_data_directory(path: &Path) -> Result<Vec<Vec<Point>>, Box<dyn Error>> {
    let mut coil_files = fs::read_dir(path)
        .expect("Error reading file")
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, io::Error>>()
        .unwrap();
    coil_files.sort();

    let mut coils = Vec::<Vec<Point>>::new();

    for coil_file in coil_files {
        let data = read_from_file(&coil_file.as_path(), usize::MAX)?;
        coils.push(data);
    }
    debug!("Read {} coils", coils.len());
    return Ok(coils);
}

pub fn compute_displacements(coil: &[Point]) -> Vec<Point> {
    coil.windows(2)
        .map(|w| w[1].get_displacement(&w[0]))
        .collect()
}

pub fn compute_all_displacements(coils: &Vec<Vec<Point>>) -> Vec<Vec<Point>> {
    coils.iter().map(|c| compute_displacements(c)).collect()
}

pub fn compute_e_roof(coil_displacements: &[Point]) -> Vec<Point> {
    coil_displacements
        .iter()
        .map(|disp| disp.get_unit_vector())
        .collect()
}

pub fn compute_all_e_roof(all_displacements: &Vec<Vec<Point>>) -> Vec<Vec<Point>> {
    all_displacements
        .iter()
        .map(|disps| compute_e_roof(disps))
        .collect()
}
