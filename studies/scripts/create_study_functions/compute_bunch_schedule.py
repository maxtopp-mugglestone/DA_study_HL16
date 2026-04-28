import json

import numpy as np


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
        number_of_bunches = 3564

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
                    _array_b2[0 : last_to_consider - number_of_bunches]
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
