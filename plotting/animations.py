import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from time import time


def animate_field_1d(
    slab,
    F,
    name,
    *F_args,
    xs=None,
    zs=None,
    fps=22,
    secs=2,
    part="real",
    color="C0",
    figsize=(11, 5),
    plot_Rhos=True,
    plot_axis=True,
    contrast=0.2,
    plot_regions=False,
    hatch="///",
    **lineargs,
):
    "Animate field F and save as name.mp4."
    plt.close("all")

    fig, ax = plt.subplots(1, figsize=figsize)
    camera = Camera(fig)
    N = int(fps * secs)

    # Make at least one of xs and zs contain data
    if xs is None and zs is None:  # Default to plotting xs at z = 0
        xs = slab.all_Xs
        zs = np.array([0])

    elif xs is not None and zs is None:
        zs = np.array([0])

    elif xs is None and zs is not None:  # Default to plotting zs at x = 0
        xs = np.array([0])

    else:
        pass

    # Give both variables at least one dimension
    try:
        len(zs)
        zs = np.array(zs)
    except TypeError:
        zs = np.array([zs])
    try:
        len(xs)
        xs = np.array(xs)
    except TypeError:
        xs = np.array([xs])

    # Check that only one of xs and zs has length one, set flags
    if len(zs) > 1 and len(xs) == 1:
        plot_Rhos = False
        plot_regions = False
        ind_vars = zs
    elif len(xs) > 1 and len(zs) == 1:
        ind_vars = xs
    elif len(zs) > 1 and len(xs) > 1:
        raise ValueError("Either xs or zs must have length 1 for 1D plot.")
    else:
        raise ValueError("At least one of xs or zs must have len > 1.")

    # Catch error if input F only has one variable
    try:
        fs = F(xs, zs, *F_args)
    except TypeError:
        fs = F(ind_vars, *F_args)

    # Animate
    for k in range(N):
        ys = fs * np.exp(-1j * k / N * 2 * np.pi)

        if part == "real":
            ys = ys.real
        elif part == "imag":
            ys = ys.imag
        elif part == "norm":
            ys = np.abs(ys)
        else:
            raise ValueError("Part must be real, imag or norm.")
        ax.plot(ind_vars, ys, color=color, **lineargs)

        if plot_Rhos:
            for Rho in slab.Rhos[1:-1]:
                ax.axvline(Rho, linestyle=":", color="lightgray")

        if plot_axis:
            ax.axhline(0, color="lightgray")
            ax.axvline(0, color="lightgray")

        if plot_regions:
            ns = slab.ns
            for i in range(len(slab.Rhos) - 1):
                n = ns[i]
                rcolor = 1 - 2 / np.pi * np.arctan(contrast * (n - 1))
                hcolor = max(
                    0.9 - 2 / np.pi * np.arctan(contrast * (n - 1)), 0
                )
                plt.axvspan(
                    slab.Rhos[i],
                    slab.Rhos[i + 1],
                    color=str(rcolor),
                    linewidth=0,
                )
                plt.axvspan(
                    slab.Rhos[i],
                    slab.Rhos[i + 1],
                    fill=None,
                    linewidth=0,
                    hatch=hatch,
                    color=str(hcolor),
                )
        camera.snap()

    animation = camera.animate(blit=True)
    animation.save(name + ".mp4", fps=fps)


def animate_field_2d(
    slab,
    F,
    filename,
    *F_args,
    ext="mp4",
    fps=22,
    secs=2,
    zmin=0,
    zmax=4,
    zref=100,
    xs=None,
    zs=None,
    levels=30,
    part="real",
    figsize=None,
    maxdim=9,
    equal=True,
    plot_Rhos=True,
    colorbar=False,
    pad=0.05,
    shrink=0.9,
    orient=None,
    colorbar_frac=0.1,
    colorbar_aspect=15,
    anchor=(0.5, 0.5),
    screen_dpi=266,
    **contourargs,
):
    "Animate field F as filled contour plot and save as name.mp4."
    plt.close("all")

    N = int(fps * secs)

    if zs is None:
        zs = np.linspace(zmin, zmax, zref)

    if xs is None:
        xs = slab.all_Xs  # doesn't duplicate endpoints

    lenzs, lenxs = np.ptp(zs), np.ptp(xs)

    if figsize is None:
        figsize = 1 / max(lenzs, lenxs) * np.array([lenxs, lenzs]) * maxdim
    else:
        figsize = figsize

    fig, ax = plt.subplots(1, figsize=figsize)

    if equal:
        ax.set_aspect("equal")

    if colorbar:
        if orient is None:
            if lenzs >= lenxs:
                orient = "horizontal"
            else:
                orient = "vertical"
        else:
            orient = orient

    camera = Camera(fig)

    Xs, Zs = np.meshgrid(xs, zs)
    Fs = F(xs, zs, *F_args)
    if part == "real":
        color_max, color_min = np.max(Fs.real), np.min(Fs.real)
    elif part == "imag":
        color_max, color_min = np.max(Fs.imag), np.min(Fs.imag)
    elif part == "norm":
        color_max, color_min = np.max(np.abs(Fs)), np.min(np.abs(Fs))
    else:
        raise ValueError("Part must be real, imag or norm.")

    # Animate
    for k in range(N):
        Ys = Fs * np.exp(-1j * k / N * 2 * np.pi)

        if part == "real":
            Ys = Ys.real
        elif part == "imag":
            Ys = Ys.imag
        elif part == "norm":
            Ys = np.abs(Ys)
        else:
            raise ValueError("Part must be real, imag or norm.")

        axmap = ax.contourf(
            Xs,
            Zs,
            Ys,
            levels=levels,
            vmin=color_min,
            vmax=color_max,
            **contourargs,
        )

        if plot_Rhos:
            for Rho in slab.Rhos[1:-1]:
                ax.axvline(Rho, linestyle=":", linewidth=1, color="k")
        if colorbar and k == 0:
            plt.colorbar(
                axmap,
                pad=pad,
                shrink=shrink,
                orientation=orient,
                anchor=anchor,
                fraction=colorbar_frac,
                aspect=colorbar_aspect,
            )
        camera.snap()

    animation = camera.animate(blit=True)
    if ext == "mp4":
        writer = "ffmpeg"
    elif ext == "gif":
        writer = "pillow"
    animation.save(
        filename + "." + ext, fps=fps, screen_dpi=screen_dpi, writer=writer
    )


def animate_field_2d_surface(
    slab,
    F,
    filename,
    *F_args,
    fps=22,
    secs=2,
    xs=None,
    zs=None,
    zmin=0,
    zmax=4,
    zref=100,
    part="real",
    figsize=(10, 5),
    show_axis=False,
    cmap="viridis",
    azim=-90,
    elev=55,
    roll=0,
    zoom=2,
    rstride=2,
    cstride=2,
    maxdim=9,
    z_lim_factor=2,
    equal=True,
    ext="mp4",
    screen_dpi=266,
    **surfaceargs,
):
    "Animate field F as 3d surface plot and save as name.mp4."
    plt.close("all")

    N = int(fps * secs)

    if zs is None:
        zs = np.linspace(zmin, zmax, zref)

    if xs is None:
        xs = slab.all_Xs  # doesn't duplicate endpoints

    lenzs, lenxs = np.ptp(zs), np.ptp(xs)

    if figsize is None:
        figsize = 1 / max(lenzs, lenxs) * np.array([lenxs, lenzs]) * maxdim
    else:
        figsize = figsize

    fig, ax = plt.subplots(
        1, figsize=figsize, subplot_kw={"projection": "3d"}
    )

    camera = Camera(fig)

    Xs, Zs = np.meshgrid(xs, zs)
    Fs = F(xs, zs, *F_args)

    if part == "real":
        color_max, color_min = np.max(Fs.real), np.min(Fs.real)
    elif part == "imag":
        color_max, color_min = np.max(Fs.imag), np.min(Fs.imag)
    elif part == "norm":
        color_max, color_min = np.max(np.abs(Fs)), np.min(np.abs(Fs))
    else:
        raise ValueError("Part must be real, imag or norm.")

    lims = (np.ptp(Xs), np.ptp(Zs), min(np.ptp(Xs), np.ptp(Zs)))
    ax.set_box_aspect(lims, zoom=zoom)
    if not show_axis:
        ax.set_axis_off()
    ax.view_init(elev, azim, roll)

    a = time()
    av_plot_time = 0
    # Animate
    for k in range(N):
        ys = Fs * np.exp(-1j * k / N * 2 * np.pi)

        if part == "real":
            ys = ys.real
        elif part == "imag":
            ys = ys.imag
        elif part == "norm":
            ys = np.abs(ys)
        else:
            raise ValueError("Part must be real, imag or norm.")

        if k == 0:
            lenys = np.ptp(ys)
            if lenys > 0:
                ax.set_zlim(-z_lim_factor * lenys, z_lim_factor * lenys)
        print("Plotting for frame ", k + 1)
        s = time()
        ax.plot_surface(
            Xs,
            Zs,
            ys,
            clip_on=False,
            cmap=cmap,
            rstride=rstride,
            cstride=cstride,
            vmin=color_min,
            vmax=color_max,
            **surfaceargs,
        )
        e = time()
        print("Done plotting.  Time taken: ", e - s)
        av_plot_time += e - s

        camera.snap()
    b = time()
    print("Done getting plots, time elapsed: ", b - a)
    plot_loop_time = b - a
    av_plot_time /= N

    animation = camera.animate(blit=True)

    if ext == "mp4":
        writer = "ffmpeg"
    elif ext == "gif":
        writer = "pillow"

    print("Starting save:")
    a = time()
    animation.save(
        filename + "." + ext, fps=fps, screen_dpi=screen_dpi, writer=writer
    )
    b = time()
    print(
        "Full plot loop time: "
        + str(plot_loop_time)
        + ".\nAvg plot time: "
        + str(av_plot_time)
        + ".\nFull save time: "
        + str(b - a)
        + ".\nAvg save time: "
        + str((b - a) / N)
    )

# ------------------------ Other Animations ------------------------------------


def radiation_mode_beta_animation(
    slab,
    name,
    beta_range=None,
    imag_type=False,
    field_type="TE",
    figsize=(11, 5),
    fps=32,
    secs=4,
    part="real",
    color="blue",
    c0=1,
    sign="+1",
    plot_Rhos=True,
    plot_axis=True,
    plot_regions=False,
    hatch="///",
    contrast=0.2,
    single_function=True,
    **lineargs,
):
    "Animate radiation field as function of beta and save as name.mp4."
    plt.close("all")

    fig, ax = plt.subplots(1, figsize=figsize)
    camera = Camera(fig)
    N = fps * secs

    if beta_range is None:
        beta_range = [-slab.K_low, slab.K_low]

    beta_range = np.linspace(beta_range[0], beta_range[1], N)

    if imag_type:
        beta_range = np.array(beta_range, dtype=complex)
        beta_range *= 1j

    # Animate
    for beta in beta_range:
        F = slab.fields(
            beta,
            c0=c0,
            field_type=field_type,
            mode_type="radiation",
            single_function=single_function,
            sign=sign,
        )

        if not single_function:
            fs = []
            for f, Xs in zip(F, slab.Xs):
                fs.append(f(Xs))
            ys = np.concatenate(fs)
            all_X = np.concatenate(slab.Xs)
        else:
            ys = F(slab.all_Xs)
            all_X = slab.all_Xs

        if part == "real":
            ys = ys.real
        elif part == "imag":
            ys = ys.imag
        elif part == "norm":
            ys = np.abs(ys)
        else:
            raise ValueError("Part must be real, imag or norm.")
        ax.plot(all_X, ys, color=color, **lineargs)
        if plot_Rhos:
            Rhos = slab.Rhos[1:-1]
            for Rho in Rhos:
                ax.axvline(Rho, linestyle=":", color="lightgray")
        if plot_axis:
            ax.axhline(0, color="lightgray")
            ax.axvline(0, color="lightgray")
        if plot_regions:
            ns = slab.ns
            for i in range(len(Rhos) - 1):
                n = ns[i]
                rcolor = 1 - 2 / np.pi * np.arctan(contrast * (n - 1))
                hcolor = max(
                    0.9 - 2 / np.pi * np.arctan(contrast * (n - 1)), 0
                )
                plt.axvspan(
                    Rhos[i], Rhos[i + 1], color=str(rcolor), linewidth=0
                )
                plt.axvspan(
                    Rhos[i],
                    Rhos[i + 1],
                    fill=None,
                    linewidth=0,
                    hatch=hatch,
                    color=str(hcolor),
                )
        camera.snap()

    animation = camera.animate(blit=True)
    animation.save(name + ".mp4", fps=fps)


def radiation_mode_Z_animation(
    slab,
    name,
    Z_range=None,
    field_type="TE",
    figsize=(11, 5),
    fps=32,
    secs=4,
    part="real",
    color="blue",
    c0=1,
    sign="+1",
    paper_method=False,
    plot_Rhos=True,
    plot_axis=True,
    plot_regions=False,
    hatch="///",
    contrast=0.2,
    single_function=True,
    **lineargs,
):
    """Animate radiation field as function of Z and save as name.mp4."""
    plt.close("all")

    fig, ax = plt.subplots(1, figsize=figsize)
    camera = Camera(fig)
    N = fps * secs

    if Z_range is None:
        Z_range = [-2 * slab.Z_evanescent, 2 * slab.Z_evanescent]

    Z_range = np.linspace(Z_range[0], Z_range[1], N)

    # Animate
    for Z in Z_range:
        F = slab.fields_Z(
            Z,
            c0=c0,
            field_type=field_type,
            mode_type="radiation",
            single_function=single_function,
            sign=sign,
            paper_method=paper_method,
        )

        if not single_function:
            fs = []
            for f, Xs in zip(F, slab.Xs):
                fs.append(f(Xs))
            ys = np.concatenate(fs)
            all_X = np.concatenate(slab.Xs)
        else:
            ys = F(slab.all_Xs)
            all_X = slab.all_Xs

        if part == "real":
            ys = ys.real
        elif part == "imag":
            ys = ys.imag
        elif part == "norm":
            ys = np.abs(ys)
        else:
            raise ValueError("Part must be real, imag or norm.")
        ax.plot(all_X, ys, color=color, **lineargs)
        if plot_Rhos:
            Rhos = slab.Rhos[1:-1]
            for Rho in Rhos:
                ax.axvline(Rho, linestyle=":", color="lightgray")
        if plot_axis:
            ax.axhline(0, color="lightgray")
            ax.axvline(0, color="lightgray")
        if plot_regions:
            ns = slab.ns
            for i in range(len(Rhos) - 1):
                n = ns[i]
                rcolor = 1 - 2 / np.pi * np.arctan(contrast * (n - 1))
                hcolor = max(
                    0.9 - 2 / np.pi * np.arctan(contrast * (n - 1)), 0
                )
                plt.axvspan(
                    Rhos[i], Rhos[i + 1], color=str(rcolor), linewidth=0
                )
                plt.axvspan(
                    Rhos[i],
                    Rhos[i + 1],
                    fill=None,
                    linewidth=0,
                    hatch=hatch,
                    color=str(hcolor),
                )
        camera.snap()

    animation = camera.animate(blit=True)
    animation.save(name + ".mp4", fps=fps)


def animate_propagation_constants(
    slab,
    rmin,
    rmax,
    imin,
    imax,
    rref=50,
    iref=50,
    levels=50,
    wlmin=1e-6,
    wlmax=2e-6,
    Nwl=5,
    filename="prop_const_animation",
    plane="Z",
    cmap="viridis",
    field_type="TE",
    mode_type="guided",
    fps=22,
    figsize=(7, 4.2),
    screen_dpi=266,
    # colorbar=True, pad=.05, shrink=.9,
    # orient='vertical', colorbar_frac=.1,
    # colorbar_aspect=15, anchor=(.5, .5),
    # color_min=None, color_max=None,
    facecolor="grey",
    grid=True,
    ext="mp4",
    **contourargs,
):
    """Animate propagation constants in complex plane as wavelength changes
    and save as name.ext."""

    if plane not in ["Psi", "beta", "Z"]:
        raise ValueError("Plane must be 'Psi', 'beta' or 'Z'.")
    plt.close("all")

    fig, ax = plt.subplots(1, figsize=figsize, screen_dpi=screen_dpi)
    camera = Camera(fig)
    ax.grid(grid, lw=0.5)
    ax.set_facecolor(facecolor)

    wls = np.linspace(wlmin, wlmax, Nwl)

    # Animate

    print("Beginning loop of getting plots")
    a = time()
    av_plot_time = 0

    for k, wl in enumerate(wls):
        slab.wl = wl

        print("Plotting for frame %i/%i" % (k + 1, Nwl))
        s = time()
        fig, ax, im = slab.determinant_plot(
            rmin,
            rmax,
            imin,
            imax,
            colorbar=False,
            rref=rref,
            iref=iref,
            levels=levels,
            plane=plane,
            mode_type=mode_type,
            field_type=field_type,
            fig=fig,
            ax=ax,
        )
        e = time()
        print("Done plotting.  Time taken: ", e - s)
        av_plot_time += e - s

        # if colorbar:
        # if k == 0:
        # cbar = plt.colorbar(im)
        # plt.colorbar(im, pad=pad, shrink=shrink,
        #              orientation=orient,
        #              anchor=anchor, fraction=colorbar_frac,
        #              aspect=colorbar_aspect)
        # fig.colorbar(im, pad=pad, shrink=shrink,
        #              orientation='vertical',
        #              anchor=anchor, fraction=colorbar_frac,
        #              aspect=colorbar_aspect, ax=[ax])
        # im.set_array(Fs.flatten())
        # cbar.boundaries = np.round(np.linspace(np.min(Fs),
        #                                        np.max(Fs),
        #                                        10), 1)
        # cbar.update_normal(im)
        camera.snap()
    b = time()
    print("Done getting plots, time elapsed: ", b - a)
    plot_loop_time = b - a
    av_plot_time /= Nwl

    if ext == "mp4":
        writer = "ffmpeg"
    elif ext == "gif":
        writer = "pillow"
    animation = camera.animate(blit=True)

    print("Starting save:")
    a = time()
    animation.save(
        filename + "." + ext, fps=fps, writer=writer, screen_dpi=screen_dpi
    )
    b = time()
    print("Done saving.\n")
    print(
        "Full plot loop time: "
        + str(plot_loop_time)
        + ".\nAvg plot time: "
        + str(av_plot_time)
        + ".\nFull save time: "
        + str(b - a)
        + ".\nAvg save time: "
        + str((b - a) / Nwl)
    )
