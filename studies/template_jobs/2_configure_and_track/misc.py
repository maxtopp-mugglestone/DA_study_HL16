# Imports
import json
import os
import pathlib as Path
import numpy as np
import xtrack as xt
from scipy.constants import c as clight
from scipy.optimize import minimize_scalar


def reformat_filling_scheme_from_lpc(filling_scheme_path, filling_scheme_path_converted):
    """
    This function is used to convert the filling scheme from the LPC to the format used in the
    xtrack library. The filling scheme from the LPC is a list of bunches for each beam, where each
    bunch is represented by a 1 in the list. The function converts this list to a list of indices
    of the filled bunches. The function also returns the indices of the filled bunches for each beam.
    """

    # Load the filling scheme directly if json
    with open(filling_scheme_path, "r") as fid:
        data = json.load(fid)

    # Take the first fill number
    fill_number = list(data["fills"].keys())[0]

    # Do the conversion (Matteo's code)
    B1 = np.zeros(3564)
    B2 = np.zeros(3564)
    l_lines = data["fills"][f"{fill_number}"]["csv"].split("\n")
    for idx, line in enumerate(l_lines):
        # First time one encounters a line with 'Slot' in it, start indexing
        if "Slot" in line:
            # B1 is initially empty
            if np.sum(B1) == 0:
                for line_2 in l_lines[idx + 1 :]:
                    l_line = line_2.split(",")
                    if len(l_line) > 1:
                        slot = l_line[1]
                        B1[int(slot)] = 1
                    else:
                        break

            elif np.sum(B2) == 0:
                for line_2 in l_lines[idx + 1 :]:
                    l_line = line_2.split(",")
                    if len(l_line) > 1:
                        slot = l_line[1]
                        B2[int(slot)] = 1
                    else:
                        break
            else:
                break

    data_json = {"beam1": [int(ii) for ii in B1], "beam2": [int(ii) for ii in B2]}

    with open(filling_scheme_path_converted, "w") as file_bool:
        json.dump(data_json, file_bool)
    return B1, B2


def load_and_check_filling_scheme(filling_scheme_path):
    """Load and check the filling scheme from a JSON file. Convert the filling scheme to the correct
    format if needed."""
    if not filling_scheme_path.endswith(".json"):
        raise ValueError("Filling scheme must be in json format")

    # Check that the converted filling scheme doesn't already exist
    filling_scheme_path_converted = filling_scheme_path.replace(".json", "_converted.json")
    if os.path.exists(filling_scheme_path_converted):
        return filling_scheme_path_converted

    with open(filling_scheme_path, "r") as fid:
        d_filling_scheme = json.load(fid)

    if "beam1" in d_filling_scheme.keys() and "beam2" in d_filling_scheme.keys():
        # If the filling scheme not already in the correct format, convert
        if "schemebeam1" in d_filling_scheme.keys() or "schemebeam2" in d_filling_scheme.keys():
            d_filling_scheme["beam1"] = d_filling_scheme["schemebeam1"]
            d_filling_scheme["beam2"] = d_filling_scheme["schemebeam2"]
            # Delete all the other keys
            d_filling_scheme = {
                k: v for k, v in d_filling_scheme.items() if k in ["beam1", "beam2"]
            }
            # Dump the dictionary back to the file
            with open(filling_scheme_path_converted, "w") as fid:
                json.dump(d_filling_scheme, fid)

            # Else, do nothing

    else:
        # One can potentially use b1_array, b2_array to scan the bunches later
        b1_array, b2_array = reformat_filling_scheme_from_lpc(
            filling_scheme_path, filling_scheme_path_converted
        )
        filling_scheme_path = filling_scheme_path_converted

    return filling_scheme_path


def _compute_LR_per_bunch(
    _array_b1, _array_b2, _B1_bunches_index, _B2_bunches_index, numberOfLRToConsider, beam="beam_1"
):
    # Reverse beam order if needed
    if beam == "beam_1":
        factor = 1
    elif beam == "beam_2":
        _array_b1, _array_b2 = _array_b2, _array_b1
        _B1_bunches_index, _B2_bunches_index = _B2_bunches_index, _B1_bunches_index
        factor = -1
    else:
        raise ValueError("beam must be either 'beam_1' or 'beam_2'")

    B2_bunches = np.array(_array_b2) == 1.0

    # Define number of LR to consider
    if isinstance(numberOfLRToConsider, int):
        numberOfLRToConsider = [numberOfLRToConsider, numberOfLRToConsider, numberOfLRToConsider]

    l_long_range_per_bunch = []
    number_of_bunches = 3564

    for n in _B1_bunches_index:
        # First check for collisions in ALICE

        # Formula for head on collision in ALICE is
        # (n + 891) mod 3564 = m
        # where n is number of bunch in B1, and m is number of bunch in B2

        # Formula for head on collision in ATLAS/CMS is
        # n = m
        # where n is number of bunch in B1, and m is number of bunch in B2

        # Formula for head on collision in LHCb is
        # (n + 2670) mod 3564 = m
        # where n is number of bunch in B1, and m is number of bunch in B2

        colide_factor_list = [891, 0, 2670]
        # i == 0 for ALICE
        # i == 1 for ATLAS and CMS
        # i == 2 for LHCB
        num_of_long_range = 0
        l_HO = [False, False, False]
        for i in range(3):
            collide_factor = colide_factor_list[i]
            m = (n + factor * collide_factor) % number_of_bunches

            # if this bunch is true, then there is head on collision
            l_HO[i] = B2_bunches[m]

            ## Check if beam 2 has bunches in range  m - numberOfLRToConsider to m + numberOfLRToConsider
            ## Also have to check if bunches wrap around from 3563 to 0 or vice versa

            bunches_ineraction_temp = np.array([])
            positions = np.array([])

            first_to_consider = m - numberOfLRToConsider[i]
            last_to_consider = m + numberOfLRToConsider[i] + 1

            if first_to_consider < 0:
                bunches_ineraction_partial = np.flatnonzero(
                    _array_b2[(number_of_bunches + first_to_consider) : (number_of_bunches)]
                )

                # This represents the relative position to the head-on bunch
                positions = np.append(positions, first_to_consider + bunches_ineraction_partial)

                # Set this varibale so later the normal syntax wihtout the wrap around checking can be used
                first_to_consider = 0

            if last_to_consider > number_of_bunches:
                bunches_ineraction_partial = np.flatnonzero(
                    _array_b2[: last_to_consider - number_of_bunches]
                )

                # This represents the relative position to the head-on bunch
                positions = np.append(positions, number_of_bunches - m + bunches_ineraction_partial)

                last_to_consider = number_of_bunches

            bunches_ineraction_partial = np.append(
                bunches_ineraction_temp,
                np.flatnonzero(_array_b2[first_to_consider:last_to_consider]),
            )

            # This represents the relative position to the head-on bunch
            positions = np.append(positions, bunches_ineraction_partial - (m - first_to_consider))

            # Substract head on collision from number of secondary collisions
            num_of_long_range_curren_ip = len(positions) - _array_b2[m]

            # Add to total number of long range collisions
            num_of_long_range += num_of_long_range_curren_ip

        # If a head-on collision is missing, discard the bunch by setting LR to 0
        if False in l_HO:
            num_of_long_range = 0

        # Add to list of long range collisions per bunch
        l_long_range_per_bunch.append(num_of_long_range)
    return l_long_range_per_bunch


def get_worst_bunch(filling_scheme_path, numberOfLRToConsider=26, beam="beam_1"):
    """
    # Adapted from https://github.com/PyCOMPLETE/FillingPatterns/blob/5f28d1a99e9a2ef7cc5c171d0cab6679946309e8/fillingpatterns/bbFunctions.py#L233
    Given a filling scheme, containing two arrays of booleans representing the trains of bunches for
    the two beams, this function returns the worst bunch for each beam, according to their collision
    schedule.
    """

    if not filling_scheme_path.endswith(".json"):
        raise ValueError("Only json filling schemes are supported")

    with open(filling_scheme_path, "r") as fid:
        filling_scheme = json.load(fid)
    # Extract booleans beam arrays
    array_b1 = np.array(filling_scheme["beam1"])
    array_b2 = np.array(filling_scheme["beam2"])

    # Get bunches index
    B1_bunches_index = np.flatnonzero(array_b1)
    B2_bunches_index = np.flatnonzero(array_b2)

    # Compute the number of long range collisions per bunch
    l_long_range_per_bunch = _compute_LR_per_bunch(
        array_b1, array_b2, B1_bunches_index, B2_bunches_index, numberOfLRToConsider, beam=beam
    )

    # Get the worst bunch for both beams
    if beam == "beam_1":
        worst_bunch = B1_bunches_index[np.argmax(l_long_range_per_bunch)]
    elif beam == "beam_2":
        worst_bunch = B2_bunches_index[np.argmax(l_long_range_per_bunch)]
    else:
        raise ValueError("beam must be either 'beam_1' or 'beam_2")

    # Need to explicitly convert to int for json serialization
    return int(worst_bunch)


# Function to generate dictionnary containing the orbit correction setup
def generate_orbit_correction_setup():
    return {
        "lhcb1": {
            "IR1 left": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.r8.b1",
                end="e.ds.l1.b1",
                vary=(
                    "corr_co_acbh14.l1b1",
                    "corr_co_acbh12.l1b1",
                    "corr_co_acbv15.l1b1",
                    "corr_co_acbv13.l1b1",
                ),
                targets=("e.ds.l1.b1",),
            ),
            "IR1 right": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.r1.b1",
                end="s.ds.l2.b1",
                vary=(
                    "corr_co_acbh13.r1b1",
                    "corr_co_acbh15.r1b1",
                    "corr_co_acbv12.r1b1",
                    "corr_co_acbv14.r1b1",
                ),
                targets=("s.ds.l2.b1",),
            ),
            "IR5 left": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.r4.b1",
                end="e.ds.l5.b1",
                vary=(
                    "corr_co_acbh14.l5b1",
                    "corr_co_acbh12.l5b1",
                    "corr_co_acbv15.l5b1",
                    "corr_co_acbv13.l5b1",
                ),
                targets=("e.ds.l5.b1",),
            ),
            "IR5 right": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.r5.b1",
                end="s.ds.l6.b1",
                vary=(
                    "corr_co_acbh13.r5b1",
                    "corr_co_acbh15.r5b1",
                    "corr_co_acbv12.r5b1",
                    "corr_co_acbv14.r5b1",
                ),
                targets=("s.ds.l6.b1",),
            ),
            "IP1": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.l1.b1",
                end="s.ds.r1.b1",
                vary=(
                    "corr_co_acbch6.l1b1",
                    "corr_co_acbcv5.l1b1",
                    "corr_co_acbch5.r1b1",
                    "corr_co_acbcv6.r1b1",
                    "corr_co_acbyhs4.l1b1",
                    "corr_co_acbyhs4.r1b1",
                    "corr_co_acbyvs4.l1b1",
                    "corr_co_acbyvs4.r1b1",
                ),
                targets=("ip1", "s.ds.r1.b1"),
            ),
            "IP2": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.l2.b1",
                end="s.ds.r2.b1",
                vary=(
                    "corr_co_acbyhs5.l2b1",
                    "corr_co_acbchs5.r2b1",
                    "corr_co_acbyvs5.l2b1",
                    "corr_co_acbcvs5.r2b1",
                    "corr_co_acbyhs4.l2b1",
                    "corr_co_acbyhs4.r2b1",
                    "corr_co_acbyvs4.l2b1",
                    "corr_co_acbyvs4.r2b1",
                ),
                targets=("ip2", "s.ds.r2.b1"),
            ),
            "IP5": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.l5.b1",
                end="s.ds.r5.b1",
                vary=(
                    "corr_co_acbch6.l5b1",
                    "corr_co_acbcv5.l5b1",
                    "corr_co_acbch5.r5b1",
                    "corr_co_acbcv6.r5b1",
                    "corr_co_acbyhs4.l5b1",
                    "corr_co_acbyhs4.r5b1",
                    "corr_co_acbyvs4.l5b1",
                    "corr_co_acbyvs4.r5b1",
                ),
                targets=("ip5", "s.ds.r5.b1"),
            ),
            "IP8": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.l8.b1",
                end="s.ds.r8.b1",
                vary=(
                    "corr_co_acbch5.l8b1",
                    "corr_co_acbyhs4.l8b1",
                    "corr_co_acbyhs4.r8b1",
                    "corr_co_acbyhs5.r8b1",
                    "corr_co_acbcvs5.l8b1",
                    "corr_co_acbyvs4.l8b1",
                    "corr_co_acbyvs4.r8b1",
                    "corr_co_acbyvs5.r8b1",
                ),
                targets=("ip8", "s.ds.r8.b1"),
            ),
        },
        "lhcb2": {
            "IR1 left": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.l1.b2",
                end="e.ds.r8.b2",
                vary=(
                    "corr_co_acbh13.l1b2",
                    "corr_co_acbh15.l1b2",
                    "corr_co_acbv12.l1b2",
                    "corr_co_acbv14.l1b2",
                ),
                targets=("e.ds.r8.b2",),
            ),
            "IR1 right": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.l2.b2",
                end="s.ds.r1.b2",
                vary=(
                    "corr_co_acbh12.r1b2",
                    "corr_co_acbh14.r1b2",
                    "corr_co_acbv13.r1b2",
                    "corr_co_acbv15.r1b2",
                ),
                targets=("s.ds.r1.b2",),
            ),
            "IR5 left": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.l5.b2",
                end="e.ds.r4.b2",
                vary=(
                    "corr_co_acbh13.l5b2",
                    "corr_co_acbh15.l5b2",
                    "corr_co_acbv12.l5b2",
                    "corr_co_acbv14.l5b2",
                ),
                targets=("e.ds.r4.b2",),
            ),
            "IR5 right": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.l6.b2",
                end="s.ds.r5.b2",
                vary=(
                    "corr_co_acbh12.r5b2",
                    "corr_co_acbh14.r5b2",
                    "corr_co_acbv13.r5b2",
                    "corr_co_acbv15.r5b2",
                ),
                targets=("s.ds.r5.b2",),
            ),
            "IP1": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.r1.b2",
                end="e.ds.l1.b2",
                vary=(
                    "corr_co_acbch6.r1b2",
                    "corr_co_acbcv5.r1b2",
                    "corr_co_acbch5.l1b2",
                    "corr_co_acbcv6.l1b2",
                    "corr_co_acbyhs4.l1b2",
                    "corr_co_acbyhs4.r1b2",
                    "corr_co_acbyvs4.l1b2",
                    "corr_co_acbyvs4.r1b2",
                ),
                targets=(
                    "ip1",
                    "e.ds.l1.b2",
                ),
            ),
            "IP2": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.r2.b2",
                end="e.ds.l2.b2",
                vary=(
                    "corr_co_acbyhs5.l2b2",
                    "corr_co_acbchs5.r2b2",
                    "corr_co_acbyvs5.l2b2",
                    "corr_co_acbcvs5.r2b2",
                    "corr_co_acbyhs4.l2b2",
                    "corr_co_acbyhs4.r2b2",
                    "corr_co_acbyvs4.l2b2",
                    "corr_co_acbyvs4.r2b2",
                ),
                targets=("ip2", "e.ds.l2.b2"),
            ),
            "IP5": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.r5.b2",
                end="e.ds.l5.b2",
                vary=(
                    "corr_co_acbch6.r5b2",
                    "corr_co_acbcv5.r5b2",
                    "corr_co_acbch5.l5b2",
                    "corr_co_acbcv6.l5b2",
                    "corr_co_acbyhs4.l5b2",
                    "corr_co_acbyhs4.r5b2",
                    "corr_co_acbyvs4.l5b2",
                    "corr_co_acbyvs4.r5b2",
                ),
                targets=(
                    "ip5",
                    "e.ds.l5.b2",
                ),
            ),
            "IP8": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.r8.b2",
                end="e.ds.l8.b2",
                vary=(
                    "corr_co_acbchs5.l8b2",
                    "corr_co_acbyhs5.r8b2",
                    "corr_co_acbcvs5.l8b2",
                    "corr_co_acbyvs5.r8b2",
                    "corr_co_acbyhs4.l8b2",
                    "corr_co_acbyhs4.r8b2",
                    "corr_co_acbyvs4.l8b2",
                    "corr_co_acbyvs4.r8b2",
                ),
                targets=(
                    "ip8",
                    "e.ds.l8.b2",
                ),
            ),
        },
    }


def compute_PU(luminosity, num_colliding_bunches, T_rev0, cross_section=81e-27):
    return luminosity / num_colliding_bunches * cross_section * T_rev0


def luminosity_leveling_ip1_5(
    collider,
    config_collider,
    config_bb,
    crab=False,
):
    # Get Twiss
    twiss_b1 = collider["lhcb1"].twiss()
    twiss_b2 = collider["lhcb2"].twiss()

    # Get the number of colliding bunches in IP1/5
    n_colliding_IP1_5 = config_collider["config_lumi_leveling_ip1_5"]["num_colliding_bunches"]

    # Get max intensity in IP1/5
    max_intensity_IP1_5 = float(
        config_collider["config_lumi_leveling_ip1_5"]["constraints"]["max_intensity"]
    )

    def compute_lumi(bunch_intensity):
        luminosity = xt.lumi.luminosity_from_twiss(  # type: ignore
            n_colliding_bunches=n_colliding_IP1_5,
            num_particles_per_bunch=bunch_intensity,
            ip_name="ip1",
            nemitt_x=config_bb["nemitt_x"],
            nemitt_y=config_bb["nemitt_y"],
            sigma_z=config_bb["sigma_z"],
            twiss_b1=twiss_b1,
            twiss_b2=twiss_b2,
            crab=crab,
        )
        return luminosity

    def f(bunch_intensity):
        luminosity = compute_lumi(bunch_intensity)
        max_PU_IP_1_5 = config_collider["config_lumi_leveling_ip1_5"]["constraints"]["max_PU"]
        target_luminosity_IP_1_5 = config_collider["config_lumi_leveling_ip1_5"]["luminosity"]
        PU = compute_PU(
            luminosity,
            n_colliding_IP1_5,
            twiss_b1["T_rev0"],
        )
        penalty_PU = max(0, (PU - max_PU_IP_1_5) * 1e35)  # in units of 1e-35
        penalty_excess_lumi = max(
            0, (luminosity - target_luminosity_IP_1_5) * 10
        )  # in units of 1e-35 if luminosity is in units of 1e34

        return abs(luminosity - target_luminosity_IP_1_5) + penalty_PU + penalty_excess_lumi

    # Do the optimization
    res = minimize_scalar(
        f,
        bounds=(
            1e10,
            max_intensity_IP1_5,
        ),
        method="bounded",
        options={"xatol": 1e7},
    )
    if not res.success:
        logging.warning("Optimization for leveling in IP 1/5 failed. Please check the constraints.")
    else:
        print(
            f"Optimization for leveling in IP 1/5 succeeded with I={res.x:.2e} particles per bunch"
        )
    return res.x


def return_fingerprint(line_name, collider):
    line = collider[line_name]

    tw = line.twiss()
    tt = line.get_table()

    det = line.get_amplitude_detuning_coefficients(a0_sigmas=0.1, a1_sigmas=0.2, a2_sigmas=0.3)

    det_table = xt.Table(
        {
            "name": np.array(list(det.keys())),
            "value": np.array([v for v in det.values()]),
        }
    )

    nl_chrom = line.get_non_linear_chromaticity(
        delta0_range=(-2e-4, 2e-4), num_delta=5, fit_order=3
    )

    out = ""

    out += f"Line: {line_name}\n"
    out += "\n"

    out += "Installed element types:\n"
    out += repr([nn for nn in sorted(list(set(tt.element_type))) if len(nn) > 0]) + "\n"
    out += "\n"

    out += f'Tunes:        Qx  = {tw["qx"]:.5f}       Qy = {tw["qy"]:.5f}\n'
    out += "Chromaticity: Q'x = " + f'{tw["dqx"]:.2f}     ' + "Q'y = " + f'{tw["dqy"]:.2f}\n'
    out += f'c_minus:      {tw["c_minus"]:.5e}\n'
    out += "\n"

    out += f'Synchrotron tune: {tw["qs"]:5e}\n'
    out += f'Slip factor:      {tw["slip_factor"]:.5e}\n'
    out += "\n"

    out += "Twiss parameters and phases at IPs:\n"
    out += (
        tw.rows["ip.*"]
        .cols["name s betx bety alfx alfy mux muy"]
        .show(output=str, max_col_width=int(1e6), digits=8)
    )
    out += "\n\n"

    out += "Dispersion at IPs:\n"
    out += (
        tw.rows["ip.*"]
        .cols["name s dx dy dpx dpy"]
        .show(output=str, max_col_width=int(1e6), digits=8)
    )
    out += "\n\n"

    out += "Crab dispersion at IPs:\n"
    out += (
        tw.rows["ip.*"]
        .cols["name s dx_zeta dy_zeta dpx_zeta dpy_zeta"]
        .show(output=str, max_col_width=int(1e6), digits=8)
    )
    out += "\n\n"

    out += "Amplitude detuning coefficients:\n"
    out += det_table.show(output=str, max_col_width=int(1e6), digits=6)
    out += "\n\n"

    out += "Non-linear chromaticity:\n"
    out += f'dnqx = {list(nl_chrom["dnqx"])}\n'
    out += f'dnqy = {list(nl_chrom["dnqy"])}\n'
    out += "\n\n"

    out += "Tunes and momentum compaction vs delta:\n"
    out += nl_chrom.show(output=str, max_col_width=int(1e6), digits=6)
    out += "\n\n"

    return out



def machine_tuning(line, config_sim,
        enable_closed_orbit_correction=False,
        enable_linear_coupling_correction=False,
        enable_tune_correction=False,
        enable_chromaticity_correction=False,
        knob_names=None,
        targets=None,
        line_co_ref=None, co_corr_config=None,
        verbose=False):
        
    # Default tolerances
    defaults = {
        'tol_qx': 1e-4,
        'tol_qy': 1e-4,
        'tol_dqx': 0.05,
        'tol_dqy': 0.05,
        'tol_cminus': 1e-4,
    }

    for key, value in defaults.items():
        if config_sim.get(key) is None:
            config_sim[key] = value

    # Correct closed orbit
    if enable_closed_orbit_correction:
        print(f'Correcting closed orbit')
        assert line_co_ref is not None
        assert co_corr_config is not None
        if isinstance(co_corr_config, (str, Path)):
            with open(co_corr_config, 'r') as fid:
                co_corr_config = json.load(fid)

        line._xmask_correct_closed_orbit(
                                reference=line_co_ref,
                                correction_config=co_corr_config)

    if enable_linear_coupling_correction:
        assert knob_names is not None
        assert 'c_minus_knob_1' in knob_names
        assert 'c_minus_knob_2' in knob_names
        # Match coupling
        print(f'Matching linear coupling')
        line.match(method = "4d", 
            verbose=verbose,
            vary=[
                xt.Vary(name=knob_names['c_minus_knob_1'],
                        limits=[-0.5e-2, 0.5e-2], step=1e-5),
                xt.Vary(name=knob_names['c_minus_knob_2'],
                        limits=[-0.5e-2, 0.5e-2], step=1e-5)],
            targets=[xt.Target('c_minus', 0, tol=config_sim['tol_cminus'])])

    # Match tune and chromaticity
    if enable_tune_correction or enable_chromaticity_correction:

        vary = []
        match_targets = []

        if enable_tune_correction:
            assert knob_names is not None
            assert 'q_knob_1' in knob_names
            assert 'q_knob_2' in knob_names
            assert targets is not None
            assert 'qx' in targets
            assert 'qy' in targets

            vary.append(xt.Vary(knob_names['q_knob_1'], step=config_sim['tol_qknobs_step']))
            vary.append(xt.Vary(knob_names['q_knob_2'], step=config_sim['tol_qknobs_step']))
            match_targets.append(xt.Target('qx', targets['qx'], tol=config_sim['tol_qx']))
            match_targets.append(xt.Target('qy', targets['qy'], tol=config_sim['tol_qy']))

        if enable_chromaticity_correction:
            assert knob_names is not None
            assert 'dq_knob_1' in knob_names
            assert 'dq_knob_2' in knob_names
            assert targets is not None
            assert 'dqx' in targets
            assert 'dqy' in targets

            vary.append(xt.Vary(knob_names['dq_knob_1'], step=1e-2))
            vary.append(xt.Vary(knob_names['dq_knob_2'], step=1e-2))
            match_targets.append(xt.Target('dqx', targets['dqx'], tol=config_sim['tol_dqx']))
            match_targets.append(xt.Target('dqy', targets['dqy'], tol=config_sim['tol_dqy']))

        print(f'Matching tune and chromaticity')
        print('Tolerances')
        print('Tolerance tune qx = ' ,config_sim['tol_qx'])
        print('Tolerance tune qy = ', config_sim['tol_qy'])
        print('Tolerance chroma dqx = ', config_sim['tol_dqx'])
        print('Tolerance chroma dqy = ', config_sim['tol_dqy'])
        print('Tolerance c_minus = ', config_sim['tol_cminus'])

        line.match(method = "4d", verbose=verbose, vary=vary, targets=match_targets)
