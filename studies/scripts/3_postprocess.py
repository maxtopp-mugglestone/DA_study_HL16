# ==================================================================================================
# --- Imports
# ==================================================================================================
# Standard library imports
import logging
import time

# Third party imports
import pandas as pd
import tree_maker


# ==================================================================================================
# --- Functions to browse simulations folder and extract relevant observables
# ==================================================================================================
# Get a given data from a dictionary with position provided as a list
def get_from_dict(dataDict, mapList):
    for k in mapList:
        dataDict = dataDict[k]
    return dataDict


def get_particles_data(root):
    l_df_output = []

    # ? Ideally node tree browsing should be done in a recursive way, but how to know in advance which
    # ? generation is being tracked?
    for node in root.generation(1):
        for node_child in node.children:
            try:
                df_output = pd.read_parquet(f"{node_child.get_abs_path()}/output_particles.parquet")
                print(node_child)
            except Exception as e:
                #print(e)
                #logging.warning(
                #    node_child.get_abs_path() + " does not have output_particles.parquet"
                #)
                continue

            # Register paths and names of the nodes
            df_output["path base collider"] = f"{node.get_abs_path()}"
            df_output["name base collider"] = f"{node.name}"
            df_output["path simulation"] = f"{node_child.get_abs_path()}"
            df_output["name simulation"] = f"{node_child.name}"

            # Add to the list
            l_df_output.append(df_output)

    return l_df_output


def reorganize_particles_data(l_df_output, dic_parameters_of_interest):
    for df_output in l_df_output:
        # Get generation configurations as dictionnaries for parameter assignation
        dic_child_collider = df_output.attrs["configuration_gen_2"]["config_collider"]
        dic_child_simulation = df_output.attrs["configuration_gen_2"]["config_simulation"]
        dic_parent_collider = df_output.attrs["configuration_gen_1"]["config_mad"]
        dic_parent_particles = df_output.attrs["configuration_gen_1"]["config_particles"]

        # Get which beam is being tracked
        df_output["beam"] = dic_child_simulation["beam"]

        # Select simulations parameters of interest
        for name_param, l_path_param in dic_parameters_of_interest.items():
            df_output[name_param] = get_from_dict(dic_child_collider, l_path_param)

        # Feel free to add more parameters of interest here (e.g. from dic_child_simulation)

    return l_df_output


def merge_and_group_by_parameters_of_interest(
    l_df_output,
    l_group_by_parameters=["beam", "name base collider", "qx", "qy"],
    only_keep_lost_particles=True,
    l_parameters_to_keep=["normalized amplitude in xy-plane", "qx", "qy", "dqx", "dqy"],
):
    # Merge the dataframes from all simulations together
    df_all_sim = pd.concat(l_df_output)

    if only_keep_lost_particles:
        # Extract the particles that were lost for DA computation
        df_all_sim = df_all_sim[df_all_sim["state"] != 1]  # Lost particles

    # Check if the dataframe is empty
    if df_all_sim.empty:
        logging.warning("No unstable particles found, the output dataframe will be empty.")

    # Group by parameters of interest
    df_grouped = df_all_sim.groupby(l_group_by_parameters)

    # Return the grouped dataframe, keeping only the minimum values of the parameters of interest
    # (should not have impact except for DA, which we want to be minimal)
    return pd.DataFrame(
        [df_grouped[parameter].min() for parameter in l_parameters_to_keep]
    ).transpose()


# ==================================================================================================
# --- Postprocess the data
# ==================================================================================================
if __name__ == "__main__":
    # Start of the script
    print("Analysis of output simulation files started")
    start = time.time()

    # Load Data
    study_name = "ext_oct_scan_end_of_collapse_round_16_dQ20"
    fix = f"/../scans/{study_name}"
    root = tree_maker.tree_from_json(fix[1:] + "/tree_maker.json")
    # Add suffix to the root node path to handle scans that are not in the root directory
    root.add_suffix(suffix=fix)

    # Get particles data
    l_df_output = get_particles_data(root)

    # Define parameters of interest
    dic_parameters_of_interest = {
        "qx": ["config_knobs_and_tuning", "qx", "lhcb1"],
        "qy": ["config_knobs_and_tuning", "qy", "lhcb1"],
        "dqx": ["config_knobs_and_tuning", "dqx", "lhcb1"],
        "dqy": ["config_knobs_and_tuning", "dqy", "lhcb1"],
        "i_oct_b1": ["config_knobs_and_tuning", "knob_settings", "i_oct_b1"],
        "i_bunch": ["config_beambeam", "mask_with_filling_pattern", "i_bunch_b1"],
        "num_particles_per_bunch": ["config_beambeam", "num_particles_per_bunch"],
    }

    # Reorganize data
    l_df_output = reorganize_particles_data(l_df_output, dic_parameters_of_interest)

    # Merge and group by parameters of interest
    l_group_by_parameters = ["beam", "name base collider", "qx", "qy", "i_oct_b1"]
    l_parameters_to_keep = [
        "normalized amplitude in xy-plane",
        "qx",
        "qy",
        "dqx",
        "dqy",
        "i_bunch",
        "i_oct_b1",
        "num_particles_per_bunch",
    ]
    only_keep_lost_particles = True
    df_final = merge_and_group_by_parameters_of_interest(
        l_df_output, l_group_by_parameters, only_keep_lost_particles, l_parameters_to_keep
    )
    print("Final dataframe for current set of simulations: ", df_final)

    # Save data and print time
    df_final.to_parquet(f"../scans/{study_name}/da.parquet")
    end = time.time()
    print(df_final["i_oct_b1"].unique())
    print("Elapsed time: ", end - start)
