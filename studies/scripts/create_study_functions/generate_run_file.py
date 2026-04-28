import yaml


def generate_run_sh(node, generation_number):
    python_command = node.root.parameters["generations"][generation_number]["job_executable"]
    return (
        "#!/bin/bash\n"
        + f"source {node.root.parameters['setup_env_script']}\n"
        + f"cd {node.get_abs_path()}\n"
        + f"python {python_command} > output_python.txt 2> error_python.txt\n"
        + "rm -rf final_* modules optics_repository optics_toolkit tools tracking_tools temp"
        " mad_collider.log __pycache__ twiss* errors fc* optics_orbit_at*\n"
    )


def generate_run_sh_htc(node, generation_number):
    python_command = node.root.parameters["generations"][generation_number]["job_executable"]
    if generation_number == 1:
        # No need to move to HTC as gen 1 is never IO intensive
        return generate_run_sh(node, generation_number)
    if generation_number == 2:
        return _generate_run_sh_htc_gen_2(node, python_command)
    if generation_number >= 3:
        print(
            f"Generation {generation_number} local htc submission is not supported yet..."
            " Submitting as for generation 1"
        )
        return generate_run_sh(node, generation_number)


def _generate_run_sh_htc_gen_2(node, python_command):
    # Get local path and abs path to gen 2
    abs_path = node.get_abs_path()
    local_path = abs_path.split("/")[-1]

    # Mutate all paths in config to be absolute
    with open(f"{abs_path}/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Get paths to mutate
    path_collider = config["config_simulation"]["collider_file"]
    path_particles = config["config_simulation"]["particle_file"]
    path_log = config["log_file"]
    new_path_collider = f"{abs_path}/{path_collider}"
    new_path_particles = f"{abs_path}/{path_particles}"
    new_path_log = f"{abs_path}/{path_log}"

    # Prepare strings for sec
    path_collider = path_collider.replace("/", "\/")
    path_particles = path_particles.replace("/", "\/")
    path_log = path_log.replace("/", "\/")
    new_path_collider = new_path_collider.replace("/", "\/")
    new_path_particles = new_path_particles.replace("/", "\/")
    new_path_log = new_path_log.replace("/", "\/")

    # Return final run script
    return (
        f"#!/bin/bash\n"
        f'source {node.root.parameters["setup_env_script"]}\n'
        # Copy config gen 1
        f"cp -f {abs_path}/../config.yaml .\n"
        # Copy config gen 2 in local path
        f"mkdir {local_path}\n"
        f"cp -f {abs_path}/config.yaml {local_path}\n"
        f"cd {local_path}\n"
        # Mutate the paths in config to be absolute
        f'sed -i "s/{path_collider}/{new_path_collider}/g" config.yaml\n'
        f'sed -i "s/{path_particles}/{new_path_particles}/g" config.yaml\n'
        f'sed -i "s/{path_log}/{new_path_log}/g" config.yaml\n'
        # Run the job
        f"python {node.get_abs_path()}/{python_command} > output_python.txt 2>"
        " error_python.txt\n"
        # Delete the config so it's not copied back
        f"rm -f ../config.yaml\n"
        # Copy back output
        f"cp -f *.txt *.parquet *.yaml {abs_path}\n"
    )
