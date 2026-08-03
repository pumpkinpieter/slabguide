import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider, Layout

# --------------------------- Interactive Plots  -------------------------------


def interactive_determinant_plot(
    slab,
    rmin,
    rmax,
    imin,
    imax,
    minwl=1e-6,
    maxwl=4e-6,
    Nwl=1000,
    rref=100,
    iref=100,
    levels=70,
    field_type="TE",
    mode_type="guided",
    plot_sdp=False,
    sdp_sign=-1,
    plane="Z",
    grid=True,
    facecolor="grey",
    figsize=(10, 6),
    width="1250px",
    plot_axis=True,
    **args,
):
    plt.close("all")
    fig, ax = plt.subplots(1, figsize=figsize)

    xs, ys = np.linspace(rmin, rmax, rref), np.linspace(imin, imax, iref)

    Xs, Ys = np.meshgrid(xs, ys)
    Cs = Xs + 1j * Ys

    def det_plot(wl, plot_sdp=plot_sdp):
        slab.wl = wl
        ax.clear()

        fs = slab.determinant(
            Cs, field_type=field_type, mode_type=mode_type, plane=plane
        )
        Vals = np.log(np.abs(fs))
        ax.contour(Xs, Ys, Vals, levels=levels, vmin=-4, vmax=9, **args)

        if plot_sdp:
            if plane == "Z":
                sdp_vals = slab.sdp(xs, sdp_sign=sdp_sign, plane=plane)
                sdp_ys = sdp_vals.imag
                msk = np.where((sdp_ys < imax) * (sdp_ys > imin))
                ax.plot(xs[msk], sdp_ys[msk], **args)
            elif plane == "Beta":
                sdp_vals = slab.sdp(ys, sdp_sign=sdp_sign, plane=plane)
                ax.plot(sdp_vals.real, sdp_vals.imag, **args)
            elif plane == "Psi":
                sdp_points = slab.sdp(ys, sdp_sign=sdp_sign, plane=plane)
                sdp_xs, sdp_ys = sdp_points.real, sdp_points.imag
                msk = np.where((sdp_xs < rmax) * (sdp_xs > rmin))
                ax.plot(sdp_xs[msk], sdp_ys[msk])
            else:
                raise ValueError("Plane must be 'Z', 'Beta' or 'Psi'.")

        ax.grid(grid)
        ax.set_facecolor(facecolor)
        if plot_axis:
            ax.axhline(0, color="k", linewidth=0.5)
            ax.axvline(0, color="k", linewidth=0.5)
        plt.title(
            "Wavelength Dependent Determinant Plot\n\
Current wavelength: %.6e\n" % (wl),
            fontsize=12,
        )

    step = (maxwl - minwl) / Nwl

    interactive_plot = interactive(
        det_plot,
        wl=FloatSlider(
            min=minwl,
            max=maxwl,
            step=step,
            value=minwl,
            readout_format=".5e",
            layout=Layout(width="90%"),
        ),
    )
    output = interactive_plot.children[-1]
    output.layout.width = "1250px"
    return interactive_plot


def interactive_spectral_integrand_1d(
    slab,
    contour,
    f0=None,
    Lx=None,
    Rx=None,
    min_x=None,
    max_x=None,
    Nx=1000,
    min_s=None,
    max_s=None,
    Ns=1000,
    min_z=0,
    max_z=15,
    Nz=1000,
    field_type="TE",
    Normalizer=None,
    class_A_only=False,
    class_B_only=False,
    figsize=(11, 4),
    width="1250px",
    part="both",
    plot_x_axis=True,
    plot_y_axis=True,
    plot_Z_evanescent=True,
    legend_fontsize=12,
    **line_args,
):
    plt.close("all")
    fig, ax = plt.subplots(1, figsize=figsize)

    Cs, contour_type = contour["Cs"], contour["contour_type"]

    if contour_type in ["real", "horizontal", "sdp"]:
        xs = Cs.real
        if max(xs) < slab.Z_evanescent.real:
            plot_Z_evanescent = False
    elif contour_type in ["imaginary", "vertical"]:
        plot_Z_evanescent = False
        xs = Cs.imag
    else:
        plot_Z_evanescent = False
        xs = contour["Ss"]

    if min_x is None:
        min_x = slab.Rhos[0]
    if max_x is None:
        max_x = slab.Rhos[-1]

    step_x = (max_x - min_x) / Nx
    step_z = (max_z - min_z) / Nz

    x_slider = FloatSlider(
        min=min_x,
        max=max_x,
        step=step_x,
        value=(max_x + min_x) / 2,
        readout_format="3f",
        layout=Layout(width="90%"),
    )

    z_slider = FloatSlider(
        min=min_z,
        max=max_z,
        step=step_z,
        value=min_z,
        readout_format=".3f",
        layout=Layout(width="90%"),
    )

    if f0 is None:
        if min_s is None:
            min_s = slab.Rhos[0]
        if max_s is None:
            max_s = slab.Rhos[-1]

        step_s = (max_s - min_s) / Ns

        s_slider = FloatSlider(
            min=min_s,
            max=max_s,
            step=step_s,
            value=(max_x + min_x) / 2,
            readout_format="3f",
            layout=Layout(width="90%"),
        )

        def int_plot(x, z, s):
            ax.clear()
            fs = slab.spectral_integrand(
                Cs,
                f0=None,
                s=s,
                x=x,
                z=z,
                Lx=Lx,
                Rx=Rx,
                field_type=field_type,
                Normalizer=Normalizer,
            )
            if part == "real":
                fs = fs.real
            elif part == "imag":
                fs = fs.imag
            elif part == "both":
                fs1 = fs.real
                fs2 = fs.imag
            elif part == "norm":
                fs = np.abs(fs)
            else:
                raise ValueError("Part must be real,imag, both, or norm.")

            if part != "both":
                ax.plot(xs, fs, label=part, **line_args)
            else:
                ax.plot(xs, fs1, label="real", **line_args)
                ax.plot(xs, fs2, label="imag", **line_args)

            if plot_x_axis:
                ax.axhline(0, color="grey", linewidth=1.25)

            if plot_y_axis:
                ax.axvline(0, color="grey", linewidth=1.25)

            if plot_Z_evanescent:
                ax.axvline(
                    slab.Z_evanescent.real,
                    color="grey",
                    linestyle=":",
                    linewidth=1.25,
                )

            plt.title(
                "$x, s, z$ Dependent 1d Spectral Integrand Plot\n\
Current x: %.3f, Current s: %.3f, Current z: %.3f" % (x, s, z),
                fontsize=12,
            )
            ax.legend(fontsize=legend_fontsize)

        interactive_plot = interactive(
            int_plot, x=x_slider, z=z_slider, s=s_slider
        )
    else:
        # if Lx is None:
        #     Lx = slab.Rhos[0]
        # if Rx is None:
        #     Rx = slab.Rhos[-1]

        if class_A_only and class_B_only:
            raise ValueError(
                "At most one of class_A_only and class_B_only \
can be set to True."
            )

        if class_A_only:
            alpha_A = slab.radiation_transform(
                Cs,
                f0=f0,
                sign="+1",
                Lx=Lx,
                Rx=Rx,
                field_type=field_type,
                Normalizer=Normalizer,
            )
            alpha_B = None

        elif class_B_only:
            alpha_B = slab.radiation_transform(
                Cs,
                f0=f0,
                sign="-1",
                Lx=Lx,
                Rx=Rx,
                field_type=field_type,
                Normalizer=Normalizer,
            )
            alpha_A = None
        else:
            alpha_A = slab.radiation_transform(
                Cs,
                f0=f0,
                sign="+1",
                Lx=Lx,
                Rx=Rx,
                field_type=field_type,
                Normalizer=Normalizer,
            )
            alpha_B = slab.radiation_transform(
                Cs,
                f0=f0,
                sign="-1",
                Lx=Lx,
                Rx=Rx,
                field_type=field_type,
                Normalizer=Normalizer,
            )

        def int_plot(x, z):
            ax.clear()
            fs = slab.spectral_integrand(
                Cs,
                f0,
                x=x,
                z=z,
                Lx=Lx,
                Rx=Rx,
                alpha_A=alpha_A,
                alpha_B=alpha_B,
                class_A_only=class_A_only,
                class_B_only=class_B_only,
                field_type=field_type,
                Normalizer=Normalizer,
            )
            if part == "real":
                fs = fs.real
            elif part == "imag":
                fs = fs.imag
            elif part == "both":
                fs1 = fs.real
                fs2 = fs.imag
            elif part == "norm":
                fs = np.abs(fs)
            else:
                raise ValueError("Part must be real,imag, both, or norm.")

            if part != "both":
                ax.plot(xs, fs, label=part, **line_args)
            else:
                ax.plot(xs, fs1, label="real", **line_args)
                ax.plot(xs, fs2, label="imag", **line_args)

            if plot_x_axis:
                ax.axhline(0, color="grey", linewidth=1.25)

            if plot_y_axis:
                ax.axvline(0, color="grey", linewidth=1.25)

            if plot_Z_evanescent:
                ax.axvline(
                    slab.Z_evanescent.real,
                    color="grey",
                    linestyle=":",
                    linewidth=1.25,
                )

            plt.title(
                "$x, z$ Dependent 1d Spectral Integrand Plot\n\
Current x: %.3f, Current z: %.3f" % (x, z),
                fontsize=12,
            )
            ax.legend(fontsize=legend_fontsize)

        interactive_plot = interactive(int_plot, x=x_slider, z=z_slider)
    output = interactive_plot.children[-1]
    output.layout.width = "1250px"
    return interactive_plot


def interactive_spectral_integrand_2d(
    slab,
    rmin,
    rmax,
    imin,
    imax,
    f0=None,
    Lx=None,
    Rx=None,
    min_x=None,
    max_x=None,
    Nx=1000,
    min_s=None,
    max_s=None,
    Ns=1000,
    min_z=0,
    max_z=15,
    Nz=1000,
    min_val=None,
    max_val=None,
    rref=100,
    iref=100,
    levels=100,
    field_type="TE",
    Normalizer=None,
    class_A_only=False,
    class_B_only=False,
    plot_sdp=True,
    sdp_sign=-1,
    plot_axis=True,
    axis_linewidth=0.7,
    axis_linecolor="k",
    grid=True,
    facecolor="grey",
    figsize=(11, 5),
    width="1250px",
    log_abs=True,
    part="norm",
    vmin=-4,
    vmax=9,
    cmap="viridis",
    plane="Z",
    **args,
):

    plt.close("all")
    fig, ax = plt.subplots(1, figsize=figsize)

    xs, ys = np.linspace(rmin, rmax, rref), np.linspace(imin, imax, iref)

    Xs, Ys = np.meshgrid(xs, ys)
    Cs = Xs + 1j * Ys

    if min_x is None:
        min_x = slab.Rhos[0]
    if max_x is None:
        max_x = slab.Rhos[-1]

    step_x = (max_x - min_x) / Nx
    step_z = (max_z - min_z) / Nz

    x_slider = FloatSlider(
        min=min_x,
        max=max_x,
        step=step_x,
        value=(max_x + min_x) / 2,
        readout_format="3f",
        layout=Layout(width="90%"),
    )

    z_slider = FloatSlider(
        min=min_z,
        max=max_z,
        step=step_z,
        value=min_z,
        readout_format=".3f",
        layout=Layout(width="90%"),
    )

    if f0 is None:
        if min_s is None:
            min_s = slab.Rhos[0]
        if max_s is None:
            max_s = slab.Rhos[-1]

        step_s = (max_s - min_s) / Ns

        s_slider = FloatSlider(
            min=min_s,
            max=max_s,
            step=step_s,
            value=(max_x + min_x) / 2,
            readout_format="3f",
            layout=Layout(width="90%"),
        )

        def int_plot(
            x, z, s, det_power=0, plot_sdp=plot_sdp, log_abs=log_abs
        ):
            ax.clear()
            fs = slab.spectral_integrand(
                Cs,
                f0=None,
                x=x,
                z=z,
                s=s,
                Lx=Lx,
                Rx=Rx,
                plane=plane,
                class_A_only=False,
                class_B_only=False,
                field_type=field_type,
                Normalizer=Normalizer,
                det_power=det_power,
            )
            if part == "real":
                fs = fs.real
            elif part == "imag":
                fs = fs.imag
            elif part == "norm":
                fs = np.abs(fs)
            else:
                raise ValueError("Part must be real, imag or norm.")
            # strange error led to this fix/style:
            # normal input to global function made vmin/vmax
            # unavailable in locally, still not sure why.
            v_min = vmin
            v_max = vmax
            if max_val is not None:
                v_max = max_val
                mask = np.where(fs > max_val)
                fs[mask] = max_val
            if min_val is not None:
                v_min = min_val
                mask = np.where(fs < min_val)
                fs[mask] = min_val
            if log_abs:
                fs = np.log(np.abs(fs))

            if plot_axis:
                if (imin <= 0) * (imax >= 0):
                    ax.axhline(
                        0, color=axis_linecolor, linewidth=axis_linewidth
                    )
                if (rmin <= 0) * (rmax >= 0):
                    ax.axvline(
                        0, color=axis_linecolor, linewidth=axis_linewidth
                    )
            ax.grid(grid)
            ax.set_facecolor(facecolor)

            ax.contour(
                Xs,
                Ys,
                fs,
                levels=levels,
                vmin=v_min,
                vmax=v_max,
                cmap=cmap,
                **args,
            )

            if plot_sdp:
                if plane == "Z":
                    sdp_vals = slab.sdp(xs, sdp_sign=sdp_sign, plane=plane)
                    sdp_xs, sdp_ys = sdp_vals.real, sdp_vals.imag
                    msk = np.where((sdp_ys < imax) * (sdp_ys > imin))
                elif plane == "Beta":
                    sdp_points = slab.sdp(
                        ys, sdp_sign=sdp_sign, plane=plane
                    )
                    sdp_xs, sdp_ys = sdp_points.real, sdp_points.imag
                    if (rmin <= slab.K0 * slab.n0) * (
                        rmax >= slab.K0 * slab.n0
                    ):
                        ax.plot(sdp_xs, sdp_ys)
                elif plane == "Psi":
                    sdp_ys = ys
                    sdp_xs = (
                        sdp_sign
                        * np.sign(sdp_ys)
                        * np.arccos(1 / np.cosh(sdp_ys))
                    )
                    msk = np.where((sdp_xs <= rmax) * (sdp_xs >= rmin))
                else:
                    raise ValueError("Plane must be 'Z', 'Beta' or 'Psi'.")

                ax.plot(sdp_xs[msk], sdp_ys[msk], **args)

            plt.title(
                "$x, s, z$ Dependent Spectral Integrand Plot\n\
Current x: %.3f, Current s: %.3f, Current z: %.3f" % (x, s, z),
                fontsize=12,
            )

        interactive_plot = interactive(
            int_plot, x=x_slider, z=z_slider, s=s_slider
        )

    else:
        # if Lx is None:
        #     Lx = slab.Rhos[0]
        # if Rx is None:
        #     Rx = slab.Rhos[-1]

        if class_A_only and class_B_only:
            raise ValueError(
                "At most one of class_A_only and class_B_only \
can be set to True."
            )

        if class_A_only:
            alpha_A = slab.radiation_transform(
                Cs,
                f0=f0,
                sign="+1",
                Lx=Lx,
                Rx=Rx,
                plane=plane,
                field_type=field_type,
                Normalizer=Normalizer,
            )
            alpha_B = None

        elif class_B_only:
            alpha_B = slab.radiation_transform(
                Cs,
                f0=f0,
                sign="-1",
                Lx=Lx,
                Rx=Rx,
                plane=plane,
                field_type=field_type,
                Normalizer=Normalizer,
            )
            alpha_A = None
        else:
            alpha_A = slab.radiation_transform(
                Cs,
                f0=f0,
                sign="+1",
                Lx=Lx,
                Rx=Rx,
                plane=plane,
                field_type=field_type,
                Normalizer=Normalizer,
            )
            alpha_B = slab.radiation_transform(
                Cs,
                f0=f0,
                sign="-1",
                Lx=Lx,
                Rx=Rx,
                plane=plane,
                field_type=field_type,
                Normalizer=Normalizer,
            )

        def int_plot(
            x, z, det_power=0, plot_sdp=plot_sdp, log_abs=log_abs
        ):
            ax.clear()
            fs = slab.spectral_integrand(
                Cs,
                f0=f0,
                x=x,
                z=z,
                Lx=Lx,
                Rx=Rx,
                plane=plane,
                alpha_A=alpha_A,
                alpha_B=alpha_B,
                class_A_only=class_A_only,
                class_B_only=class_B_only,
                field_type=field_type,
                Normalizer=Normalizer,
                det_power=det_power,
            )
            if part == "real":
                fs = fs.real
            elif part == "imag":
                fs = fs.imag
            elif part == "norm":
                fs = np.abs(fs)
            else:
                raise ValueError("Part must be real, imag or norm.")
            if log_abs:
                fs = np.log(np.abs(fs))

            if plot_axis:
                if (imin <= 0) * (imax >= 0):
                    ax.axhline(
                        0, color=axis_linecolor, linewidth=axis_linewidth
                    )
                if (rmin <= 0) * (rmax >= 0):
                    ax.axvline(
                        0, color=axis_linecolor, linewidth=axis_linewidth
                    )

            ax.grid(grid)
            ax.set_facecolor(facecolor)

            ax.contour(
                Xs,
                Ys,
                fs,
                levels=levels,
                # vmin=vmin, vmax=vmax,
                cmap=cmap,
                **args,
            )

            if plot_sdp:
                if plane == "Z":
                    sdp_vals = slab.sdp(xs, sdp_sign=sdp_sign, plane=plane)
                    sdp_xs, sdp_ys = sdp_vals.real, sdp_vals.imag
                    msk = np.where((sdp_ys < imax) * (sdp_ys > imin))
                elif plane == "Beta":
                    sdp_points = slab.sdp(
                        ys, sdp_sign=sdp_sign, plane=plane
                    )
                    sdp_xs, sdp_ys = sdp_points.real, sdp_points.imag
                    if (rmin <= slab.K0 * slab.n0) * (
                        rmax >= slab.K0 * slab.n0
                    ):
                        ax.plot(sdp_xs, sdp_ys)
                elif plane == "Psi":
                    sdp_ys = ys
                    sdp_xs = (
                        sdp_sign
                        * np.sign(sdp_ys)
                        * np.arccos(1 / np.cosh(sdp_ys))
                    )
                    msk = np.where((sdp_xs <= rmax) * (sdp_xs >= rmin))
                else:
                    raise ValueError("Plane must be 'Z', 'Beta' or 'Psi'.")
                ax.plot(sdp_xs[msk], sdp_ys[msk], **args)

            plt.title(
                "$x, z$ Dependent Spectral Integrand Plot\n\
Current x: %.3f, Current z: %.3f" % (x, z),
                fontsize=12,
            )

        interactive_plot = interactive(int_plot, x=x_slider, z=z_slider)
    output = interactive_plot.children[-1]
    output.layout.width = "1250px"
    return interactive_plot


def interactive_radiation_mode_plot(
    slab,
    minZ=0.001,
    maxZ=1,
    NZ=200,
    xs=None,
    field_type="TE",
    sign="+1",
    minwl=None,
    maxwl=None,
    Nwl=300,
    figsize=(11, 5),
    width="1250px",
    part="real",
    plane="Z",
    method="paper",
    ylims=None,
    plot_Rhos=True,
    **line_args,
):
    plt.close("all")
    fig, ax = plt.subplots(1, figsize=figsize)

    if part not in ["real", "imag", "norm"]:
        raise ValueError("Part must be 'real', 'imag', or 'norm'.")
    if xs is None:
        xs = slab.all_Xs

    if minwl is None or maxwl is None:
        minwl, maxwl = slab.wl, slab.wl

    methods = slab.radiation_mode_normalization_methods
    if method not in methods:
        raise ValueError("Method must be one of: " + str(methods))
    Normalizer = slab.normalizer(method=method)

    def F_plot(Z, wl):
        ax.clear()
        slab.wl = wl

        if sign == "both":
            F1 = slab.fields(
                Z,
                plane=plane,
                mode_type="radiation",
                field_type=field_type,
                Normalizer=Normalizer,
                sign="1",
            )

            F2 = slab.fields(
                Z,
                plane=plane,
                mode_type="radiation",
                field_type=field_type,
                Normalizer=Normalizer,
                sign="-1",
            )

            ys1, ys2 = F1(xs), F2(xs)
            if part == "real":
                ys1, ys2 = ys1.real, ys2.real
            elif part == "imag":
                ys1, ys2 = ys1.imag, ys2.imag
            else:
                ys1, ys2 = np.abs(ys1), np.abs(ys2)
            ax.plot(xs, ys1, **line_args)
            ax.plot(xs, ys2, **line_args)

        else:
            F = slab.fields(
                Z,
                plane=plane,
                mode_type="radiation",
                field_type=field_type,
                Normalizer=Normalizer,
                sign=sign,
            )
            ys = F(xs)

            if part == "real":
                ys = ys.real
            elif part == "imag":
                ys = ys.imag
            else:
                ys = np.abs(ys)
            ax.plot(xs, ys, **line_args)

        ax.axhline(0, color="grey", linewidth=1.25)
        ax.axvline(0, color="grey", linewidth=1.25)

        if ylims is not None:
            ax.set_ylim(ylims[:])

        if plot_Rhos:
            for Rho in slab.Rhos[1:-1]:
                ax.axvline(Rho, linestyle=":", linewidth=1, color="grey")

        plt.title(
            "Z Dependent Radiation Mode Plot\n$Z = %.4e$\n" % Z,
            fontsize=12,
        )

    Zstep = (maxZ - minZ) / NZ
    Zslider = FloatSlider(
        min=minZ,
        max=maxZ,
        step=Zstep,
        value=minZ,
        readout_format=".3e",
        layout=Layout(width="90%"),
    )

    wlstep = (maxwl - minwl) / Nwl
    wlslider = FloatSlider(
        min=minwl,
        max=maxwl,
        value=minwl,
        step=wlstep,
        readout_format=".3e",
        layout=Layout(width="90%"),
    )

    interactive_plot = interactive(F_plot, Z=Zslider, wl=wlslider)
    output = interactive_plot.children[-1]
    output.layout.width = "1250px"

    return interactive_plot
