use clap::Parser;
use log::{debug, info, trace};
use mpi::traits::{Communicator, CommunicatorCollectives, Root};
use std::{
    fs::{self},
    path::Path,
};

use bs_solctra_rs::{args, point, simulation, utils};

fn main() {
    env_logger::init();
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let world_size = world.size();
    let rank = world.rank();
    let processor = mpi::environment::processor_name().unwrap();

    if rank == 0 {
        info!("Starting BS-Solctra");
        info!("Total ranks: {}", world_size);
    }
    trace!("Rank: {}, processor: {}", rank, processor);
    let args = args::Args::parse();
    if rank == 0 {
        trace!("{:?}", args);
    }
    let output_dir = Path::new(&args.output);

    if rank == 0 {
        match fs::exists(output_dir) {
            Ok(true) => info!("Output path: {} already exists", args.output),
            Ok(false) => utils::create_directory(output_dir),
            Err(err) => panic!("Error querying path: {}", err),
        }
        info!("Reading particles from file {}", args.particles_file);
    }
    let max_particles = args.num_particles;
    let particles_per_rank = max_particles / world_size as usize;
    let mut local_particles = vec![
        point::Point {
            x: 0.0,
            y: 0.0,
            z: 0.0
        };
        particles_per_rank
    ];
    if rank == 0 {
        let particles = match point::read_from_file(Path::new(&args.particles_file), max_particles)
        {
            Ok(particles) => particles,
            Err(err) => panic!("Error: {}", err),
        };
        world
            .process_at_rank(0)
            .scatter_into_root(particles.as_slice(), local_particles.as_mut_slice());
    } else {
        world
            .process_at_rank(0)
            .scatter_into(local_particles.as_mut_slice());
    }

    world.barrier();

    debug!(
        "Rank: {}, local particles length: {}",
        rank,
        local_particles.len()
    );

    trace!("Rank {}, {:?}", rank, local_particles);

    if rank == 0 {
        info!("Reading coil data from directory: {}", &args.resource_path);
    }
    let coils = match simulation::read_coil_data_directory(Path::new(&args.resource_path)) {
        Ok(coils) => coils,
        Err(err) => panic!("Error: {}", err),
    };
    if rank == 0 {
        info!("Computing displacements");
    }
    let displacements = simulation::compute_all_displacements(&coils);
    if rank == 0 {
        info!("Computing e_roof");
    }
    let e_roof = simulation::compute_all_e_roof(&displacements);
    if rank == 0 {
        debug!("Total e_roof: {}", e_roof.len());
        trace!("{:?}", e_roof);

        info!("Computing simulation")
    }

    world.barrier();
    let t_start = mpi::time();
    simulation::simulate_particles(
        local_particles.as_mut_slice(),
        args.steps,
        args.step_size,
        &coils,
        &displacements,
        &e_roof,
        output_dir,
        args.write_frequency,
        rank,
    );
    world.barrier();
    let t_end = mpi::time();
    if rank == 0 {
        info!("Finished simulation");
        info!("Simulation time: {}", t_end - t_start);
    }
}
