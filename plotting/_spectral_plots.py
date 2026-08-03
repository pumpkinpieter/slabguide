
    def spectral_integrand_plot_1d(
        self,
        contour,
        s=None,
        f0=None,
        Lx=None,
        Rx=None,
        x0=0,
        z0=0,
        field_type="TE",
        Normalizer=None,
        class_A_only=False,
        class_B_only=False,
        figsize=(11, 4),
        part="both",
        plot_x_axis=True,
        plot_y_axis=True,
        plot_Z_evanescent=True,
        legend_fontsize=12,
        **line_args
        ):
 
        if (f0 is None and s is None) or (f0 is not None and s is not None):
            raise ValueError(
                "Exactly one of either input field 'f0' or source \
point 's' must be provided."
            )
        plt.close("all")

        fig, ax = plt.subplots(1, figsize=figsize)
        
        Cs, contour_type = contour["Cs"], contour["contour_type"]

        if contour_type in ["real", "horizontal", "sdp"]:
            xs = Cs.real
            if max(xs) < self.Z_evanescent.real:
                plot_Z_evanescent = False
        elif contour_type in ["imaginary", "vertical"]:
            plot_Z_evanescent = False
            xs = Cs.imag
        else:
            plot_Z_evanescent = False
            xs = contour["Ss"]

        fs = self.spectral_integrand(
            Cs,
            f0=f0,
            s=s,
            x=x0,
            z=z0,
            Lx=Lx,
            Rx=Rx,
            field_type=field_type,
            Normalizer=Normalizer,
            class_A_only=class_A_only,
            class_B_only=class_B_only,
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
            raise ValueError("Part must be real, imag, both, or norm.")

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
                self.Z_evanescent.real,
                color="grey",
                linestyle=":",
                linewidth=1.25,
            )
        if f0 is None:
            title = "1d Spectral Integrand Plot\n\
 x: %.3f, s: %.3f, z: %.3f" % (x0, s, z0)
        else:
            title = "1d Spectral Integrand Plot\n\
 x: %.3f, z: %.3f" % (x0, z0)            
        plt.title(title, fontsize=12)
        ax.legend(fontsize=legend_fontsize)

    
    def spectral_integrand_plot_2d(
        self,
        rmin,
        rmax,
        imin,
        imax,
        rref=100,
        iref=100,
        levels=100,
        f0=None,
        s=None,
        x=0,
        z=0,
        Lx=None,
        Rx=None,
        field_type="TE",
        class_A_only=False,
        class_B_only=False,
        plot_sdp=True,
        plot_axis=True,
        axis_linewidth=0.7,
        axis_linecolor="k",
        sdp_sign=-1,
        grid=True,
        facecolor="grey",
        figsize=(11, 5),
        width="1250px",
        log_abs=True,
        part="norm",
        max_z=None,
        min_z=None,
        vmin=-4,
        vmax=9,
        cmap="viridis",
        plane="Z",
        Normalizer=None,
        colorbar=True,
        pad=0.02,
        shrink=1,
        colorbar_frac=0.05,
        colorbar_aspect=30,
        anchor=(0.5, 1),
        det_power=0,
        **args,
    ):
        plt.close("all")
        fig, ax = plt.subplots(1, figsize=figsize)
        xs, ys = np.linspace(rmin, rmax, rref), np.linspace(imin, imax, iref)

        Xs, Ys = np.meshgrid(xs, ys)
        Cs = Xs + 1j * Ys

        if class_A_only and class_B_only:
            raise ValueError(
                "At most one of class_A_only and class_B_only can \
be set to True."
            )

        if class_A_only:
            alpha_A = self.radiation_transform(
                Cs,
                f0=f0,
                s=s,
                sign=1,
                Lx=Lx,
                Rx=Rx,
                plane=plane,
                field_type=field_type,
                Normalizer=Normalizer,
            )
            alpha_B = None

        elif class_B_only:
            alpha_B = self.radiation_transform(
                Cs,
                f0=f0,
                s=s,
                sign=-1,
                Lx=Lx,
                Rx=Rx,
                plane=plane,
                field_type=field_type,
                Normalizer=Normalizer,
            )
            alpha_A = None
        else:
            alpha_A = self.radiation_transform(
                Cs,
                f0=f0,
                s=s,
                sign=1,
                Lx=Lx,
                Rx=Rx,
                plane=plane,
                field_type=field_type,
                Normalizer=Normalizer,
            )
            alpha_B = self.radiation_transform(
                Cs,
                f0=f0,
                s=s,
                sign=-1,
                Lx=Lx,
                Rx=Rx,
                plane=plane,
                field_type=field_type,
                Normalizer=Normalizer,
            )

        fs = self.spectral_integrand(
            Cs,
            f0=f0,
            s=s,
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
        if max_z is not None:
            vmax = max_z
            mask = np.where(fs > max_z)
            fs[mask] = max_z
        if min_z is not None:
            vmin = min_z
            mask = np.where(fs < min_z)
            fs[mask] = min_z
        if log_abs:
            fs = np.log(np.abs(fs))
        axmap = ax.contour(
            Xs, Ys, fs, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap, **args
        )

        if plot_sdp:
            if plane == "Z":
                sdp_vals = self.sdp(xs, sdp_sign=sdp_sign, plane=plane)
                sdp_xs, sdp_ys = sdp_vals.real, sdp_vals.imag
                msk = np.where((sdp_ys <= imax) * (sdp_ys >= imin))
                ax.plot(sdp_xs[msk], sdp_ys[msk], **args)
            elif plane == "Beta":
                sdp_points = self.sdp(ys, sdp_sign=sdp_sign, plane=plane)
                sdp_xs, sdp_ys = sdp_points.real, sdp_points.imag
                if (rmin <= self.K0 * self.n0) * (rmax >= self.K0 * self.n0):
                    ax.plot(sdp_xs, sdp_ys)
            elif plane == "Psi":
                sdp_ys = ys
                sdp_xs = (
                    sdp_sign * np.sign(sdp_ys) * np.arccos(1 / np.cosh(sdp_ys))
                )
                msk = np.where((sdp_xs <= rmax) * (sdp_xs >= rmin))
                ax.plot(sdp_xs[msk], sdp_ys[msk], **args)
            else:
                raise ValueError("Plane must be 'Z', 'Beta' or 'Psi'.")
                
        if plot_axis:
            if (imin <= 0) * (imax >= 0):
                ax.axhline(0, color=axis_linecolor, linewidth=axis_linewidth)
            if (rmin <= 0) * (rmax >= 0):
                ax.axvline(0, color=axis_linecolor, linewidth=axis_linewidth)
                
        ax.grid(grid)
        ax.set_facecolor(facecolor)
        if s is not None:
            plt.title(
                "Spectral Integrand Plot for x = %.2f, s = %.2f, \
z = %.2f" % (x, s, z),
                fontsize=12,
            )
        else:
            plt.title(
                "Spectral Integrand Plot for x = %.2f, z = %.2f" % (x, z),
                fontsize=12,
            )
        if colorbar:
            plt.colorbar(
                axmap,
                pad=pad,
                shrink=shrink,
                orientation="vertical",
                anchor=anchor,
                fraction=colorbar_frac,
                aspect=colorbar_aspect,
                ax=ax,
            )

    def spectral_integrand_surface_plot(
        self,
        rmin,
        rmax,
        imin,
        imax,
        rref=100,
        iref=100,
        f0=None,
        s=None,
        x=0,
        z=0,
        Lx=None,
        Rx=None,
        field_type="TE",
        class_A_only=False,
        class_B_only=False,
        max_z=None,
        min_z=None,
        det_power=0,
        width="1250px",
        plane="Z",
        log_abs=True,
        part="norm",
        cmap="viridis",
        Normalizer=None,
        colorbar_aspect=30,
        figsize=(10, 5),
        show_axis=False,
        azim=-90,
        elev=55,
        roll=0,
        zoom=2,
        rstride=2,
        cstride=2,
        z_lim_factor=2,
        colorbar=False,
        pad=0.15,
        shrink=0.85,
        colorbar_frac=0.15,
        anchor=(0.5, 0.5),
        orient="vertical",
        **surfaceargs,
    ):
        """Plot field F as filled contour plot.

        If no xs are provided, plots entire region defined by SlabExact class.

        Note that we avoid applying meshgrid to xs and zs until after calling
        function. Since field is a product of single function of xs and
        single function of zs it would be wasteful to call it on the meshed
        arrays."""
        plt.close("all")

        fig, ax = plt.subplots(
            1, figsize=figsize, subplot_kw={"projection": "3d"}
        )
        xs, ys = np.linspace(rmin, rmax, rref), np.linspace(imin, imax, iref)

        Xs, Ys = np.meshgrid(xs, ys)
        Cs = Xs + 1j * Ys

        if class_A_only and class_B_only:
            raise ValueError(
                "At most one of class_A_only and class_B_only can \
be set to True."
            )

        if class_A_only:
            alpha_A = self.radiation_transform(
                Cs,
                f0=f0,
                s=s,
                sign=1,
                Lx=Lx,
                Rx=Rx,
                plane=plane,
                field_type=field_type,
                Normalizer=Normalizer,
            )
            alpha_B = None

        elif class_B_only:
            alpha_B = self.radiation_transform(
                Cs,
                f0=f0,
                s=s,
                sign=-1,
                Lx=Lx,
                Rx=Rx,
                plane=plane,
                field_type=field_type,
                Normalizer=Normalizer,
            )
            alpha_A = None
        else:
            alpha_A = self.radiation_transform(
                Cs,
                f0=f0,
                s=s,
                sign=1,
                Lx=Lx,
                Rx=Rx,
                plane=plane,
                field_type=field_type,
                Normalizer=Normalizer,
            )
            alpha_B = self.radiation_transform(
                Cs,
                f0=f0,
                s=s,
                sign=-1,
                Lx=Lx,
                Rx=Rx,
                plane=plane,
                field_type=field_type,
                Normalizer=Normalizer,
            )

        fs = self.spectral_integrand(
            Cs,
            f0=f0,
            s=s,
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
        if max_z is not None:
            mask = np.where(fs > max_z)
            fs[mask] = max_z
        if min_z is not None:
            mask = np.where(fs < min_z)
            fs[mask] = min_z
        if log_abs:
            fs = np.log(np.abs(fs))

        axmap = ax.plot_surface(
            Xs,
            Ys,
            fs,
            clip_on=False,
            cmap=cmap,
            rstride=rstride,
            cstride=cstride,
            **surfaceargs,
        )

        lims = (np.ptp(Xs), np.ptp(Ys), min(np.ptp(Xs), np.ptp(Ys)))
        fs_width = np.ptp(fs)
        ax.set_zlim(-z_lim_factor * fs_width, z_lim_factor * fs_width)
        ax.set_box_aspect(lims, zoom=zoom)
        if not show_axis:
            ax.set_axis_off()
        ax.view_init(elev, azim, roll)

        if colorbar:
            plt.colorbar(
                axmap,
                pad=pad,
                shrink=shrink,
                orientation=orient,
                anchor=anchor,
                fraction=colorbar_frac,
            )
        return fig, ax

