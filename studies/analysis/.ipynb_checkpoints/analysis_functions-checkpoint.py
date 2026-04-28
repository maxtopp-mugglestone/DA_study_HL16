import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline
import numpy as np
import yaml
from scipy.ndimage.filters import gaussian_filter


def apply_heatmap_style():
    plt.style.use("ggplot")
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["font.family"] = "STIXGeneral"
    # Not italized latex
    matplotlib.rcParams["mathtext.default"] = "regular"
    matplotlib.rcParams["font.weight"] = "light"
    matplotlib_inline.backend_inline.set_matplotlib_formats("retina")


# Function to convert floats to scientific latex format
def latex_float(f):
    float_str = "{0:.3g}".format(f)
    if "e" not in float_str:
        return float_str
    base, exponent = float_str.split("e")
    return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))


def load_config(config_path):
    # Read configuration file
    with open(config_path, "r") as fid:
        config = yaml.safe_load(fid)
    return config


def get_title_from_conf(
    conf_mad,
    conf_collider=None,
    type_crossing=None,
    betx=None,
    bety=None,
    Nb=True,
    levelling="",
    CC=False,
    display_intensity=True,
    PU=True,
):
    # LHC version
    LHC_version = "HL-LHC v1.6"

    # Energy
    energy_value = float(conf_mad["beam_config"]["lhcb1"]["beam_energy_tot"]) / 1000
    energy = f"$E = {{{energy_value:.1f}}}$ $TeV$"

    if conf_collider is not None:
        # Levelling
        levelling = levelling
        if levelling != "":
            levelling += " ."

        # Bunch number
        bunch_number_value = conf_collider["config_beambeam"]["mask_with_filling_pattern"][
            "i_bunch_b1"
        ]
        bunch_number = f"Bunch {bunch_number_value}"

        # Crab cavities
        if CC:
            if "on_crab1" in conf_collider["config_knobs_and_tuning"]["knob_settings"]:
                if (
                    conf_collider["config_knobs_and_tuning"]["knob_settings"]["on_crab1"]
                    is not None
                ):
                    CC_value = conf_collider["config_knobs_and_tuning"]["knob_settings"]["on_crab1"]
                    crab_cavities = f"$CC = {{{CC_value:.1f}}}$ $\mu rad$. "  # type: ignore
                else:
                    crab_cavities = "CC OFF. "
            else:
                crab_cavities = "NO CC. "
        else:
            crab_cavities = ""

        # Bunch intensity
        if Nb:
            try:
                bunch_intensity_value = conf_collider["config_beambeam"][
                    "num_particles_per_bunch_after_optimization"
                ]
            except Exception:
                bunch_intensity_value = conf_collider["config_beambeam"]["num_particles_per_bunch"]
            bunch_intensity = f"$N_b \simeq ${latex_float(float(bunch_intensity_value))} ppb, "  # type: ignore
        else:
            bunch_intensity = ""

        try:
            luminosity_value_1 = conf_collider["config_beambeam"][
                "luminosity_ip1_after_optimization"
            ]
            luminosity_value_5 = conf_collider["config_beambeam"][
                "luminosity_ip5_after_optimization"
            ]
            luminosity_value_1_5 = np.mean([luminosity_value_1, luminosity_value_5])
            luminosity_value_2 = conf_collider["config_beambeam"][
                "luminosity_ip2_after_optimization"
            ]
            luminosity_value_8 = conf_collider["config_beambeam"][
                "luminosity_ip8_after_optimization"
            ]
        except:  # noqa: E722
            print("Luminosity not found in config, setting to None")
            luminosity_value_1_5 = None
            luminosity_value_2 = None
            luminosity_value_8 = None
        if luminosity_value_1_5 is not None:
            luminosity_1_5 = (
                f"$L_{{1/5}} = ${latex_float(float(luminosity_value_1_5))}" + "cm$^{-2}$s$^{-1}$, "
            )
            luminosity_2 = (
                f"$L_{{2}} = ${latex_float(float(luminosity_value_2))}" + "cm$^{-2}$s$^{-1}$, "
            )
            luminosity_8 = (
                f"$L_{{8}} = ${latex_float(float(luminosity_value_8))}" + "cm$^{-2}$s$^{-1}$"
            )
        else:
            luminosity_1_5 = ""
            luminosity_2 = ""
            luminosity_8 = ""

        if PU:
            try:
                PU_value_1 = conf_collider["config_beambeam"]["Pile-up_ip1_5_after_optimization"]
                PU_value_5 = conf_collider["config_beambeam"]["Pile-up_ip1_5_after_optimization"]
            except:  # noqa: E722
                try:
                    PU_value_1 = conf_collider["config_beambeam"]["Pile-up_ip1_after_optimization"]
                    PU_value_5 = conf_collider["config_beambeam"]["Pile-up_ip5_after_optimization"]
                except:  # noqa: E722
                    PU_value_1 = None
                    PU_value_5 = None

            try:
                PU_value_2 = conf_collider["config_beambeam"]["Pile-up_ip2_after_optimization"]
                PU_value_8 = conf_collider["config_beambeam"]["Pile-up_ip8_after_optimization"]
            except:  # noqa: E722
                PU_value_2 = None
                PU_value_8 = None
            if PU_value_1 is not None:
                PU_1_5 = f"$PU_{{1/5}} = ${latex_float(float(PU_value_1))}, "
                PU_2 = f"$PU_{{2}} = ${latex_float(float(PU_value_2))}, "
                PU_8 = f"$PU_{{8}} = ${latex_float(float(PU_value_8))}" + ""
            else:
                PU_1_5 = ""
                PU_2 = ""
                PU_8 = ""

        # Beta star # ! Manually encoded for now
        if "flathv" in conf_mad["optics_file"]:
            bet1 = r"$\beta^{*}_{y,1}$"
            bet2 = r"$\beta^{*}_{x,1}$"
        # If betas are given, we always define betx first, whatever the crossing
        elif "flatvh" in conf_mad["optics_file"] or (betx is not None and bety is not None):
            bet1 = r"$\beta^{*}_{x,1}$"
            bet2 = r"$\beta^{*}_{y,1}$"
        if betx is not None and bety is not None:
            beta = bet1 + f"$= {{{betx}}}$" + " m, " + bet2 + f"$= {{{bety}}}$" + " m"

        # Crossing angle at IP1/5
        if "flathv" in conf_mad["optics_file"] or type_crossing == "flathv":
            phi_1 = r"$\Phi/2_{1(H)}$"
            phi_5 = r"$\Phi/2_{5(V)}$"
        elif "flatvh" in conf_mad["optics_file"] or type_crossing == "flatvh":
            phi_1 = r"$\Phi/2_{1(V)}$"
            phi_5 = r"$\Phi/2_{5(H)}$"
        else:
            phi_1 = r"$\Phi/2_{1(H)}$"
            phi_5 = r"$\Phi/2_{5(V)}$"
        # else:
        #     raise ValueError("Optics configuration not automatized yet")
        xing_value_IP1 = conf_collider["config_knobs_and_tuning"]["knob_settings"]["on_x1"]
        xing_IP1 = phi_1 + f"$= {{{xing_value_IP1:.0f}}}$" + " $\mu rad$"

        xing_value_IP5 = conf_collider["config_knobs_and_tuning"]["knob_settings"]["on_x5"]
        xing_IP5 = phi_5 + f"$= {{{xing_value_IP5:.0f}}}$" + " $\mu rad$"

        # Bunch length
        bunch_length_value = conf_collider["config_beambeam"]["sigma_z"] * 100
        bunch_length = f"$\sigma_{{z}} = {{{bunch_length_value}}}$ $cm$"

        # Crosing angle at IP8
        xing_value_IP8h = conf_collider["config_knobs_and_tuning"]["knob_settings"]["on_x8h"]
        xing_value_IP8v = conf_collider["config_knobs_and_tuning"]["knob_settings"]["on_x8v"]
        if xing_value_IP8v != 0 and xing_value_IP8h == 0:
            xing_IP8 = r"$\Phi/2_{8,V}$" + f"$= {{{xing_value_IP8v:.0f}}}$ $\mu rad$"
        elif xing_value_IP8v == 0 and xing_value_IP8h != 0:
            xing_IP8 = r"$\Phi/2_{8,H}$" + f"$= {{{xing_value_IP8h:.0f}}}$ $\mu rad$"
        else:
            raise ValueError("Optics configuration not automatized yet")

        # Crosing angle at IP2
        try:
            xing_value_IP2h = conf_collider["config_knobs_and_tuning"]["knob_settings"]["on_x2h"]
            xing_value_IP2v = conf_collider["config_knobs_and_tuning"]["knob_settings"]["on_x2v"]
        except:  # noqa: E722
            xing_value_IP2h = 0
            xing_value_IP2v = conf_collider["config_knobs_and_tuning"]["knob_settings"]["on_x2"]
        if xing_value_IP2v != 0 and xing_value_IP2h == 0:
            xing_IP2 = r"$\Phi/2_{2,V}$" + f"$= {{{xing_value_IP2v:.0f}}}$ $\mu rad$"
        elif xing_value_IP8v == 0 and xing_value_IP8h != 0:
            xing_IP2 = r"$\Phi/2_{2,H}$" + f"$= {{{xing_value_IP2h:.0f}}}$ $\mu rad$"
        else:
            raise ValueError("Optics configuration not automatized yet")

        # Polarity IP 2 and 8
        polarity_value_IP2 = conf_collider["config_knobs_and_tuning"]["knob_settings"][
            "on_alice_normalized"
        ]
        polarity_value_IP8 = conf_collider["config_knobs_and_tuning"]["knob_settings"][
            "on_lhcb_normalized"
        ]
        polarity = f"$polarity$ $IP_{{2/8}} = {{{polarity_value_IP2}}}/{{{polarity_value_IP8}}}$"

        # Normalized emittance
        emittance_value = round(conf_collider["config_beambeam"]["nemitt_x"] / 1e-6, 2)
        emittance = f"$\epsilon_{{n}} = {{{emittance_value}}}$ $\mu m$"

        # Chromaticity
        chroma_value = conf_collider["config_knobs_and_tuning"]["dqx"]["lhcb1"]
        chroma = r"$Q'$" + f"$= {{{chroma_value}}}$"

        # Intensity
        if display_intensity:
            intensity_value = conf_collider["config_knobs_and_tuning"]["knob_settings"]["i_oct_b1"]
            intensity = f"$I_{{MO}} = {{{intensity_value}}}$ $A$, "
        else:
            intensity = ""

        # Linear coupling
        coupling_value = conf_collider["config_knobs_and_tuning"]["delta_cmr"]
        coupling = f"$C^- = {{{coupling_value}}}$"

        # Filling scheme
        filling_scheme_value = conf_collider["config_beambeam"]["mask_with_filling_pattern"][
            "pattern_fname"
        ].split("filling_scheme/")[1]
        if "12inj" in filling_scheme_value:
            filling_scheme_value = filling_scheme_value.split("12inj")[0] + "12inj"
        filling_scheme = f"{filling_scheme_value}"
        title = (
            LHC_version
            + ". "
            + energy
            + ". "
            + levelling
            + crab_cavities
            + bunch_intensity
            + "\n"
            + luminosity_1_5
            + luminosity_2
            + luminosity_8
            + "\n"
            + PU_1_5
            # + PU_2
            # + PU_8
            + beta
            + ", "
            + polarity
            + "\n"
            + xing_IP1
            + ", "
            + xing_IP5
            + ", "
            + xing_IP2
            + ", "
            + xing_IP8
            + "\n"
            + bunch_length
            + ", "
            + emittance
            + ", "
            + chroma
            + ", "
            + intensity
            + coupling
            + "\n"
            + filling_scheme
            + ". "
            + bunch_number
            + "."
        )
    else:
        title = LHC_version + ". " + energy + ". "
    return title


def plot_heatmap(
    df_to_plot,
    study_name,
    link=None,
    plot_contours=True,
    conf_mad=None,
    conf_collider=None,
    type_crossing=None,
    betx=None,
    bety=None,
    Nb=True,
    levelling="",
    CC=False,
    xlabel="Horizontal tune " + r"$Q_x$",
    ylabel="Vertical tune " + r"$Q_y$",
    symmetric=True,
    mask_lower_triangle=True,
    mask_upper_triangle=False,
    plot_diagonal_lines=True,
    xaxis_ticks_on_top=True,
    title=None,
    add_vline=None,
    display_intensity=True,
    vmin=4.5,
    vmax=7.5,
    extended_diagonal=False,
    green_contour=6,
):
    # Get numpy array from dataframe
    data_array = df_to_plot.to_numpy()

    # Define colormap and set NaNs to white
    cmap = matplotlib.cm.get_cmap("coolwarm_r", 50)
    cmap.set_bad("w")

    # Build heatmap, with inverted y axis
    fig, ax = plt.subplots()
    extent = df_to_plot.columns[0], df_to_plot.columns[-1], df_to_plot.index[-1], df_to_plot.index[0],
    print(extent)
    #im = ax.imshow(data_array, cmap=cmap, vmin=vmin, vmax=vmax)
    im = ax.imshow(data_array, cmap=cmap, vmin=vmin, vmax=vmax, extent = extent, aspect = 'auto')
    #im = ax.pcolormesh(df_to_plot.columns, df_to_plot.index, data_array, cmap = cmap, vmin = vmin, vmax = vmax)
    
    aspect = np.abs((extent[1]-extent[0])/(extent[3]-extent[2]))/1.5
    print(aspect)
    ax.set_aspect(aspect)
    ax.invert_yaxis()
    #print(breaker)

    # Show all ticks and label them with the respective list entries
    #ax.set_xticks(np.arange(len(df_to_plot.columns))[::4], labels=df_to_plot.columns[::4])
    #ax.set_xticks(np.arange(len(df_to_plot.columns))[::4], labels=df_to_plot.columns[::4])
    
    #ax.set_yticks(np.arange(len(df_to_plot.index))[::2], labels=df_to_plot.index[::2])
    #ticks = np.arange(-600, 301, 50)
    '''
    # Loop over data dimensions and create text annotations.
    for i in range(len(df_to_plot.index)):
        for j in range(len(df_to_plot.columns)):
            if data_array[i, j] >= vmax:
                val = r"$\geq $" + str(vmax)
            elif data_array[i, j] <= vmin:
                val = r"$\leq $" + str(vmin)
            else:
                val = f"{data_array[i, j]:.1f}"
            text = ax.text(j, i, val, ha="center", va="center", color="white", fontsize=4)
    '''
    # Smooth data for contours
    # make the matrix symmetric by replacing the lower triangle with the upper triangle
    data_smoothed = np.copy(data_array)
    data_smoothed[np.isnan(data_array)] = 0
    if symmetric and not extended_diagonal:
        data_smoothed = data_smoothed + data_smoothed.T - np.diag(data_array.diagonal())
    elif symmetric:
        try:
            # sum the upper and lower triangle, but not the intersection of the two matrices
            intersection = np.zeros_like(data_smoothed)
            for x in range(data_smoothed.shape[0]):
                for y in range(data_smoothed.shape[1]):
                    if np.min((data_smoothed[x, y], data_smoothed[y, x])) == 0.0:
                        intersection[x, y] = 0.0
                    else:
                        intersection[x, y] = data_smoothed[y, x]
            data_smoothed = data_smoothed + data_smoothed.T - intersection
        except:
            print("Did not manage to smooth properly")
    data_smoothed = gaussian_filter(data_smoothed, 0.7)
    # Mask the lower triangle of the smoothed matrix
    if not extended_diagonal and mask_lower_triangle:
        mask = np.tri(data_smoothed.shape[0], k=-1)
        mx = np.ma.masked_array(data_smoothed, mask=mask.T)
    elif not extended_diagonal and mask_upper_triangle:
        mask = np.tri(data_smoothed.shape[0], k=-1)
        mx = np.ma.masked_array(data_smoothed, mask=mask)
    elif extended_diagonal:
        try:
            mask = np.tri(data_smoothed.shape[0], k=-5)
            mx = np.ma.masked_array(data_smoothed, mask=mask.T)
        except:
            print("Did not manage to mask properly")
            mx = data_smoothed
    else:
        mx = data_smoothed

    # Plot contours if requested
    if plot_contours:
        CSS = ax.contour(
            df_to_plot.columns,#np.arange(0.5, data_array.shape[1]),
            df_to_plot.index,#np.arange(0.5, data_array.shape[0]),
            mx,
            colors="black",
            levels=list(np.arange(3, green_contour, 0.5))
            + list(np.arange(green_contour + 0.5, 10, 0.5)),
            linewidths=0.2,
        )
        ax.clabel(CSS, inline=True, fontsize=6)
        CS2 = ax.contour(
            df_to_plot.columns,#np.arange(0.5, data_array.shape[1]),
            df_to_plot.index,#np.arange(0.5, data_array.shape[0]),
            #np.arange(0.5, data_array.shape[1]),
            #np.arange(0.5, data_array.shape[0]),
            mx,
            colors="green",
            levels=[green_contour],
            linewidths=1,
        )
        ax.clabel(CS2, inline=1, fontsize=6)
        upper = np.array([l.split(', ') for l in '''-286.4010989010989, 62.32194082397004
        -260.4395604395604, 62.322155056179774
        -234.8901098901099, 62.32195730337079
        -210.5769230769231, 62.32195730337079
        -186.26373626373626, 62.322006741573034
        -131.04395604395606, 62.3216277153558
        -110.02747252747253, 62.32136404494382
        -11.950549450549488, 62.32030936329588
        14.835164835164846, 62.32027640449438
        48.626373626373606, 62.31953483146067
        65.1098901098901, 62.31902397003745
        90.24725274725273, 62.31872734082397
        160.30219780219778, 62.31828239700374
        189.97252747252747, 62.31834831460674
        215.52197802197804, 62.318068164794006
        241.89560439560432, 62.318134082397
        301.2362637362637, 62.31747490636704
        312.3626373626373, 62.31716179775281'''.split('\n')], dtype=float)

        lower = np.array([l.split(', ') for l in '''-285.989010989011, 62.31750786516854
-261.2637362637363, 62.31727715355805
-236.12637362637363, 62.31727715355805
-211.40109890109892, 62.316964044943816
-182.967032967033, 62.31702996254681
-160.71428571428572, 62.31686516853932
-111.26373626373626, 62.316123595505616
-85.71428571428572, 62.316387265917605
-61.40109890109892, 62.31620599250936
-26.785714285714278, 62.3154808988764
-10.714285714285722, 62.31480524344569
14.423076923076906, 62.314574531835206
39.560439560439534, 62.315447940074904
89.83516483516479, 62.315563295880146
136.8131868131868, 62.31516779026217
162.77472527472526, 62.31508539325843
190.7967032967033, 62.315447940074904
217.1703296703297, 62.31538202247191
238.1868131868132, 62.315118352059926
261.2637362637363, 62.31538202247191
288.8736263736264, 62.315151310861424
311.53846153846155, 62.31566217228465'''.split('\n')], dtype=float)

        #plt.plot(*upper.T, c='m', lw=2)
        #plt.plot(*lower.T, c='m', lw=2)
        #plt.xlim(right=300)
        #plt.ylim(62.305, 62.329)

    if plot_diagonal_lines:
        # ! Diagonal lines must be plotted after the contour lines, because of bug in matplotlib
        # ! Careful, depending on how the tunes were defined, may be shifted by 1
        # Diagonal lines
        if extended_diagonal:
            ax.plot([0, 1000], [5, 1005], color="tab:blue", linestyle="--", linewidth=1)
            ax.plot([0, 1000], [-5, 995], color="tab:blue", linestyle="--", linewidth=1)
            ax.plot([0, 1000], [0, 1000], color="black", linestyle="--", linewidth=1)
        else:
            ax.plot([0, 1000], [1, 1001], color="tab:blue", linestyle="--", linewidth=1)
            ax.plot([0, 1000], [-9, 991], color="tab:blue", linestyle="--", linewidth=1)
            ax.plot([0, 1000], [-4, 996], color="black", linestyle="--", linewidth=1)

    # Define title and axis labels
    if title is None:
        ax.set_title(
            get_title_from_conf(
                conf_mad,
                conf_collider,
                type_crossing=type_crossing,
                betx=betx,
                bety=bety,
                Nb=Nb,
                levelling=levelling,
                CC=CC,
                display_intensity=display_intensity,
            ),
            fontsize=10,
        )
    else:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #print(test)
    #ax.set_xlim(0 - 0.5, data_array.shape[1] - 0.5)
    #ax.set_ylim(0 - 0.5, data_array.shape[0] - 0.5)
    #ax.set_xlim(extent[0:2])
    #ax.set_ylim(extent[2:])

    # Ticks on top
    if xaxis_ticks_on_top:
        ax.xaxis.tick_top()
    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=-30,
        rotation_mode="anchor",  # , ha="left"
    )  # , rotation_mode="anchor")
    # ax.tick_params(axis='x', which='major', pad=5)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.026, pad=0.04)
    cbar.ax.set_ylabel("Minimum DA (" + r"$\sigma$" + ")", rotation=90, va="bottom", labelpad=15)
    plt.grid(visible=None)

    if add_vline is not None:
        plt.axvline(add_vline, color="black", linestyle="--", linewidth=1)
        plt.text(add_vline, 25, r"Bunch intensity $\simeq \beta^*_{2023} = 0.3/0.3$", fontsize=8)

    plt.savefig("plots/output_" + study_name + ".pdf", bbox_inches="tight")
    plt.show()