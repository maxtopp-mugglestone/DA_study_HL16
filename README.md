# Dynamics aperture study template

This repository contains a template that allows users to compute the dynamic aperture of HL LHC 1.6 using Xsuite (latest version 0.48.4)

This is based on code from the following repositories:
https://github.com/skostogl/example_DA_study/
https://github.com/ColasDroin/DA_IPAC_2024/
https://github.com/aradosla/example_DA_study/
https://github.com/andreafornara/DA_study_template_afornara/


## Quick use guide

Update scripts/config.yaml. Change eos_python to point to your python environment. A packaged python environment using conda pack is included as envs.tar.gz

Modify scripts/1_create_study_$$.py to match your requirements for the study. Two examples are included, representing an octupole and a tune scan. Some issues were encountered with relative paths for the optics files - I suggest uploading optics files to eos so a static absolute path can be provided. Check the paths for the optics files and base sequence. Change the study name to a unique string.

Modify scripts/2_run_jobs.py and scripts/3_postprocess.py to change the study name at the bottom to what you set in the previous step.

When running, first run your 1_create_study.py, then after it completes, run your 2_run_jobs.py. Wait for that to complete generation 1 (distribution/collider building) before launching 2_run_jobs.py again. Finally, after this completes, run 3_postprocess.py (studies/analysis should contain an example plotting notebook)

## Known issues

A small number of instances fail in the beam beam configuration steps. This is a closed orbit search failure within a twiss() function call from inside Xsuite's multiline legacy functions. The 6d closed orbit search is less robust in the current version of Xsuite compared to the version used in 2024, but more accurate in describing certain physical phenomena. Modifying this to a twiss4d() call would fix the problem, but as this multiline legacy support is deprecated, this is a low priority issue (could be fixed if needed by installing xsuite development version and modifying the call manually).



## Parameters that can be scanned

At the moment, all the collider parameters can be scanned without requiring extensive scripts modifications. This includes (but is not limited to):

- intensity (```num_particles_per_bunch```)
- crossing-angle (```on_x1, on_x5```)
- tune (```qx, qy```)
- chromaticity (```dqx, dqy```)
- octupole current (```i_oct_b1, i_oct_b2```)
- bunch being tracked (```i_bunch_b1, i_bunch_b2```)

At generation 1, the base collider is built with a default set of parameters for the optics (which are explicitely set in ```1_create_study.py```). At generation 2, the base collider is tailored to the parameters being scanned. That is, the tune and chroma are matched, the luminosity leveling is computed (if leveling is required), and the beam-beam lenses are configured.

It should be relatively easy to accomodate the scripts for other parameters.

## License

This repository is licensed under the MIT license. Please refer to the [LICENSE](LICENSE) file for more information.

# 
add in Manifest.in of xmak: recursive-include xmask/lhc *
