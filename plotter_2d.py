import csv
import os
import matplotlib
matplotlib.use("Agg")   # must be before pyplot

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

class Plotter2D:
    """
    Class to handle 2D real-time plotting of multiple fields:
    - phi (order parameter)
    - rho (density)
    - vorticity
    - velocity vectors
    """
    def __init__(self, script_dir, script_filename, images_subdir,
                 total_iterations, filename_padding_width, debug_log,
                 PLOTREALTIME=True, save_frames=False):
        self.PLOTREALTIME = PLOTREALTIME
        self.save_frames = save_frames

        self.script_dir = script_dir
        self.SCRIPT_FILENAME = script_filename
        self.IMAGES_SUBDIR = images_subdir
        self.TOTAL_ITERATIONS = total_iterations
        self.FILENAME_PADDING_WIDTH = filename_padding_width
        self.debug_log = debug_log        

        if PLOTREALTIME:
            if self.save_frames:
                self.frames_dir = os.path.join(self.script_dir, "frames2D")
                os.makedirs(self.frames_dir, exist_ok=True)

            # Initialize figure and axes (your previous GridSpec setup)
            self.fig_rt = plt.figure(figsize=(8, 12))
            gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,1,0.8], hspace=0.2)
            self.ax_phi = self.fig_rt.add_subplot(gs[0])
            self.ax_rho = self.fig_rt.add_subplot(gs[1])
            self.ax_vort = self.fig_rt.add_subplot(gs[2])
            self.ax_vel = self.fig_rt.add_subplot(gs[3])
            self.fig_rt.subplots_adjust(top=0.98, bottom=0.03, left=0.05, right=0.95)

            # Placeholders
            self.im_phi = None
            self.im_rho = None
            self.im_vort = None
            self.im_vel = None


    def update(self, iteration, _phi, rho, u_ckl, arrow_spacing=2):
        """
        Update the real-time 2D plots and save frames for movie.
        Args:
            iteration: current iteration
            _phi: 2D array (202, 52)
            rho: 2D array (202, 52)
            u_ckl: 3D array (2, 202, 52)
            arrow_spacing: spacing for quiver arrows
        """
        if not self.PLOTREALTIME or (iteration % 10 != 0 and iteration != self.TOTAL_ITERATIONS - 1):
            return

        # -------------------------------
        # Compute fields
        # -------------------------------
        ux = u_ckl[0]
        uy = u_ckl[1]
        vorticity = (np.roll(ux, -1, 0) - np.roll(ux, 1, 0)) - (np.roll(uy, -1, 1) - np.roll(uy, 1, 1))

        # Swap axes and flip for correct orientation
        phi_plot = np.flipud(np.swapaxes(_phi, 0, 1))
        rho_plot = np.flipud(np.swapaxes(rho, 0, 1))
        vort_plot = np.flipud(np.swapaxes(vorticity, 0, 1))
        ux_plot = np.flipud(np.swapaxes(ux, 0, 1))
        uy_plot = np.flipud(np.swapaxes(uy, 0, 1))

        # -------------------------------
        # Velocity quivers
        # -------------------------------
        vel_mag = np.sqrt(ux_plot**2 + uy_plot**2)
        alpha = np.arctan2(uy_plot, ux_plot)

        # Mask: zero magnitude or unchanged direction
        if hasattr(self, 'prev_alpha') and self.prev_alpha is not None:
            mask_dir = np.abs(alpha - self.prev_alpha) > 1e-6
        else:
            mask_dir = np.ones_like(alpha, dtype=bool)
        mask = (vel_mag > 1e-12) & mask_dir
        self.prev_alpha = alpha.copy()

        # Scale arrows for visibility
        vel_max = np.max(vel_mag)
        scale_factor = 0.9 * min(ux_plot.shape) / (vel_max + 1e-12)
        u_quiv = ux_plot[::arrow_spacing, ::arrow_spacing] * mask[::arrow_spacing, ::arrow_spacing] * scale_factor
        v_quiv = uy_plot[::arrow_spacing, ::arrow_spacing] * mask[::arrow_spacing, ::arrow_spacing] * scale_factor

        # Colorize arrows by magnitude
        vel_norm = vel_mag / (vel_max + 1e-12)
        vel_colors = cm.jet(vel_norm)
        vel_colors_quiv = vel_colors[::arrow_spacing, ::arrow_spacing].reshape(-1, 4)

        x = np.arange(0, ux_plot.shape[1], arrow_spacing)
        y = np.arange(0, ux_plot.shape[0], arrow_spacing)

        # -------------------------------
        # Initialize or update images
        # -------------------------------
        if self.im_phi is None:
            # Phi
            self.im_phi = self.ax_phi.imshow(phi_plot, cmap="bwr", origin='lower', vmin=_phi.min(), vmax=_phi.max())
            self._setup_ax(self.ax_phi, "Phi - order parameter")
            self._add_colorbar(self.im_phi, self.ax_phi, invert=True, shrink=0.9)

            # Rho (invert so liquid at bottom)
            self.im_rho = self.ax_rho.imshow(rho_plot, cmap="viridis_r", origin='lower', vmin=rho.min(), vmax=rho.max())
            self._setup_ax(self.ax_rho, "Rho - density distribution")
            self._add_colorbar(self.im_rho, self.ax_rho, invert=True, shrink=0.9)

            # Vorticity
            vmin_v, vmax_v = np.min(vorticity), np.max(vorticity)
            if vmin_v == vmax_v: vmax_v += 1e-10
            self.im_vort = self.ax_vort.imshow(vort_plot, cmap="bwr", origin='lower', vmin=vmin_v, vmax=vmax_v)
            self._setup_ax(self.ax_vort, "Vorticity - 2D curl of u_ckl")
            self._add_colorbar(self.im_vort, self.ax_vort, invert=False, shrink=0.9)

            # Velocity vectors
            self.im_vel = self.ax_vel.quiver(
                x, y, u_quiv, v_quiv,
                pivot='middle', color=vel_colors_quiv,
                scale_units='xy', angles='xy', scale=1.0
            )
            self._setup_ax(self.ax_vel, "Velocity vectors in nodes")

        else:
            # Update existing images
            self.im_phi.set_data(phi_plot)
            self.im_rho.set_data(rho_plot)
            vmin_v, vmax_v = np.min(vorticity), np.max(vorticity)
            if vmin_v == vmax_v: vmax_v += 1e-10
            self.im_vort.set_data(vort_plot)
            self.im_vort.set_clim(vmin_v, vmax_v)
            self.im_vel.set_UVC(u_quiv, v_quiv)
            self.im_vel.set_color(vel_colors_quiv)

        # -------------------------------
        # Refresh figure (compact)
        # -------------------------------
        self.fig_rt.subplots_adjust(hspace=0.15, top=0.97, bottom=0.03)
        self.fig_rt.canvas.draw()

        # -------------------------------
        # Save frame for movie
        # -------------------------------
        if not hasattr(self, 'frames_dir'):
            self.frames_dir = os.path.join(self.script_dir, "frames2D")
            os.makedirs(self.frames_dir, exist_ok=True)
        if self.save_frames:
            frame_filename = os.path.join(self.frames_dir, f"frame_{iteration:0{self.FILENAME_PADDING_WIDTH}d}.png")
            self.fig_rt.savefig(frame_filename, dpi=150, bbox_inches='tight')


    # -------------------------------
    # Helper methods
    # -------------------------------
    @staticmethod
    def _setup_ax(ax, title):
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.invert_yaxis()

    #@staticmethod
    def _add_colorbar(self, im, ax, invert=False, shrink=1.0):
        """
        Add a colorbar that matches the axes height and optionally invert it.

        Args:
            im: Image object (AxesImage)
            ax: Corresponding axis
            invert: If True, invert the colorbar (large -> bottom, small -> top)
            shrink: Fraction to shrink colorbar height
        """
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.02)
        cbar = plt.colorbar(im, cax=cax, shrink=shrink)
        if invert:
            cbar.ax.invert_yaxis()


    #amplitude_plot
    def amplitude_plot(
        self,
        ax1,
        u_full_range,
        listIterations,  # not used directly, but kept for compatibility
        axis,
        xlabel,
        ylabel,
        title,
        nx,
        ny,
        angle=45,
        font_size=10,
    ):
        """
        Plots amplitude profiles for multiple iterations.
        Automatically sparsifies the legend to avoid crowding.
        """
        # Sorted list of iteration numbers (keys in u_full_range)
        key_iterations = sorted(u_full_range.keys())
        n_items = len(key_iterations)

        if n_items == 0:
            return

        # === Decide legend sparsity ===
        if n_items <= 8:
            step = 1
        elif n_items <= 20:
            step = 2
        elif n_items <= 40:
            step = 4
        else:
            step = max(5, n_items // 8)  # Aim for roughly 8–10 legend entries max

        # Determine which iterations should have visible labels
        show_labels = set()
        for i in range(0, len(key_iterations), step):
            show_labels.add(key_iterations[i])
        # Always include first and last (in case step skips them)
        show_labels.add(key_iterations[0])
        show_labels.add(key_iterations[-1])

        n_visible = len(show_labels)

        # Adjust number of columns in legend based on visible entries
        ncol = max(1, min(4, (n_visible + 6) // 7))  # Rough heuristic: ~7 items per column

        # ======================
        # === Standalone plot ===
        # ======================
        fig_standalone = plt.figure(figsize=(10, 6))
        ax_standalone = fig_standalone.add_subplot(111)

        for iteration in key_iterations:
            u = u_full_range[iteration][nx, 1:ny + 1]
            label = f"t={iteration}" if iteration in show_labels else None
            ax_standalone.plot(u, axis, label=label)

        ax_standalone.grid(True)
        ax_standalone.set_xlabel(xlabel)
        ax_standalone.set_ylabel(ylabel)
        ax_standalone.set_title(title)
        ax_standalone.set_ylim(-1, 51)
        ax_standalone.set_yticks(np.arange(0, 51, 10))

        ax_standalone.legend(
            ncol=ncol,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fontsize=font_size,
            frameon=True,
        )

        plt.setp(ax_standalone.get_xticklabels(), rotation=angle, ha="right", fontsize=font_size)
        ax_standalone.tick_params(axis="x", labelsize=font_size)
        ax_standalone.margins(x=0, y=0)
        x_min, x_max = ax_standalone.get_xlim()
        ax_standalone.set_xlim(x_min, x_max + 0.1 * (x_max - x_min))

        # Save standalone
        filename = f"{title.replace(' ', '_')}_{max(key_iterations[-1], self.TOTAL_ITERATIONS):0{self.FILENAME_PADDING_WIDTH}d}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)
        fig_standalone.savefig(save_path, dpi=300, bbox_inches="tight")
        self.debug_log("INIT", "Saved amplitude plot: %s", save_path)
        plt.close(fig_standalone)

        # ======================
        # === Subplot version ===
        # ======================
        for iteration in key_iterations:
            u = u_full_range[iteration][nx, 1:ny + 1]
            label = f"t={iteration}" if iteration in show_labels else None
            ax1.plot(u, axis, label=label)

        ax1.grid(True)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(title)
        ax1.set_ylim(-1, 51)
        ax1.set_yticks(np.arange(0, 51, 10))

        ax1.legend(
            ncol=ncol,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fontsize=font_size,
            frameon=True,
        )

        plt.setp(ax1.get_xticklabels(), rotation=angle, ha="right", fontsize=font_size)
        ax1.tick_params(axis="x", labelsize=font_size)
        ax1.margins(x=0, y=0)
        x_min, x_max = ax1.get_xlim()
        ax1.set_xlim(x_min, x_max + 0.1 * (x_max - x_min))


    # Density profile
    def density_profile(self, ax1, den_eq, nx, ny, iteration=0):
        # Calculate y-positions for 1/3, 1/2, and 2/3 of the channel height
        y_third = ny // 3  # 1/3 * ymax
        y_half = ny // 2   # 1/2 * ymax
        y_two_thirds = 2 * ny // 3  # 2/3 * ymax

        # Plot density along x for each y-position
        ax1.plot(den_eq[1:nx, y_third], label=f"y = {y_third} (~1/3 ymax)")
        ax1.plot(den_eq[1:nx, y_half], label=f"y = {y_half} (~1/2 ymax)")
        ax1.plot(den_eq[1:nx, y_two_thirds], label=f"y = {y_two_thirds} (~2/3 ymax)")

        ax1.set_xlabel("x-axis")
        ax1.set_ylabel("Density [rho]")
        ax1.margins(x=0, y=0)
        ax1.set_title("Longitudinal density profile")
        ax1.yaxis.tick_left()
        ax1.yaxis.set_label_position("left")
        ax1.set_xlim(0, nx)
        ax1.set_xticks(np.linspace(0, nx, 5))
        ax1.grid()

        ax1.legend(loc='best')  # Add legend to distinguish the three lines

        # Save in same directory as the script
        filename = f"density_profile_{iteration:0{self.FILENAME_PADDING_WIDTH}d}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)

        ax1.figure.savefig(save_path, dpi=300, bbox_inches='tight')
        self.debug_log('INIT', 'Saved density profile: %s', save_path)


    def density_profile_transition(self, ax1, den_eq, x__position, y_position, nx, ny, granularity, iteration=0):
        # Fixed x-position at Xn/2
        #x_fixed = nx // 2  # Xn/2 = 100 for nx = 200

        # Define y-coordinates for the plot
        y_coords = np.arange(1, ny)  # y = 1 to ny-1 (1 to 50 for ny = 51)

        # Plot density (rho) on x-axis, y-coordinate on y-axis
        ax1.plot(den_eq[x__position, 1:ny], y_coords, label=f"x = {x__position}")

        ax1.set_xlabel("Density [rho]")
        ax1.set_ylabel("y-axis")
        ax1.margins(x=0, y=0)
        ax1.set_title("Vertical density profile at x = Xn/2")
        ax1.yaxis.tick_left()
        ax1.yaxis.set_label_position("left")
        ax1.set_xlim(-1, 52)  # Density range: rho_G - 2 = -1, rho_L + 2 = 52
        ax1.set_xticks(np.linspace(-1, 52, 5))
        ax1.set_ylim(y_position - granularity, y_position + granularity) 
        ax1.set_yticks(np.linspace(y_position - granularity, y_position + granularity, 5))
        ax1.grid()

        ax1.legend(loc='best')  # Add legend

        # Save in same directory as the script
        filename = f"density_profile_transition_{iteration:0{self.FILENAME_PADDING_WIDTH}d}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)

        ax1.figure.savefig(save_path, dpi=300, bbox_inches='tight')
        self.debug_log('INIT', 'Saved density profile transition: %s', save_path)


    # Plot density profiles saved at given iterations
    def density_profiles(self, ax, density_slices, x__position, nx, ny, iteration=0):
        """Compact overlay of density slices with right legend."""
        n_items = len(density_slices)
        ncol = max(1, min(3, (n_items + 3) // 4))  # 4–5 items per column, max 3 cols
        if not density_slices: return
        y_coords = np.arange(0, ny + 2)
        iterations, slices = zip(*density_slices)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for idx, (it, rho_slice) in enumerate(zip(iterations, slices)):
            ax.plot(rho_slice, y_coords, color=colors[idx % 10], label=f"Iter {it}")
        ax.set(xlabel="Density [rho]", ylabel="y-axis", title=f"Evolution at x={x__position}",
            xlim=(-1, 52), xticks=np.linspace(-1, 52, 5), ylim=(0, ny + 1), yticks=np.linspace(0, ny + 1, 6))
        ax.grid()
        ax.legend(
            ncol=ncol,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            fontsize='small',
            frameon=True
        )
        #ax.figure.tight_layout()

        # Save in same directory as the script
        filename = f"density_profiles_{iteration:0{self.FILENAME_PADDING_WIDTH}d}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)

        ax.figure.savefig(save_path, dpi=300, bbox_inches='tight')
        self.debug_log('INIT', 'Saved density profiles: %s', save_path)


    #2D Density map
    def density_mapExt(self, ax, full_range, min, max, title, iteration):
        self.debug_log('FIELD', 'Debug density_map: min=%.6f, max=%.6f, rho_out=%.6f, rho_in=%.6f', 
            np.min(full_range), np.max(full_range), min, max)
        im = ax.imshow(full_range.T, interpolation='nearest', origin='lower', cmap='viridis', vmin=min, vmax=max)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title(title)
        ax.margins(x=0, y=0)
        ax.set_aspect('auto')
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #cbar.set_ticks([min, (min + max) / 2, max])  
        cbar.set_ticks(np.linspace(min, max, 3)) 

        # Save in same directory as the script
        filename = f"density_mapExt_{iteration:0{self.FILENAME_PADDING_WIDTH}d}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)

        ax.figure.savefig(save_path, dpi=300, bbox_inches='tight')
        self.debug_log('INIT', 'Saved density_mapExt: %s', save_path)


    def density_map_standalone(self, full_range, min_val, max_val, title, iteration):
        """
        Save a density map plot to a PNG file using a canvas (no GUI window).

        Args:
            full_range: 2D numpy array of density values (Ny, Nx).
            min_val: Lower bound for colorbar (float).
            max_val: Upper bound for colorbar (float).
            title: Plot title (str).
            iteration: Current iteration number (int).

        Returns:
            None (saves PNG).
        """
        self.debug_log('FIELD', 'Debug density_map: min=%.6f, max=%.6f', 
                np.min(full_range), np.max(full_range))

        fig = Figure(figsize=(6, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        if not np.isfinite(min_val) or not np.isfinite(max_val) or min_val == max_val:
            min_val = 0.0
            max_val = 1.0
        im = ax.imshow(full_range.T, interpolation='nearest', origin='lower',
                    cmap='viridis', vmin=min_val, vmax=max_val)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title(title)
        ax.margins(x=0, y=0)
        ax.set_aspect('auto')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(min_val, max_val, 3))  # min, mid, max
        cbar.set_label("Density")

        filename = f"density_map_standalone_{iteration:0{self.FILENAME_PADDING_WIDTH}d}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)
        canvas.print_figure(save_path, dpi=300, bbox_inches='tight')

        self.debug_log('INIT', 'Saved density_map_standalone: %s', save_path)


    def phi_profile(self, phi_array, plot_name, iteration=0):
        phi_2d = phi_array[np.newaxis, :]  # 1 row
        phi_min, phi_max = np.min(phi_2d), np.max(phi_2d)

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(
            phi_2d.T,  # transpose so y runs horizontally
            interpolation='nearest',
            origin='lower',
            cmap='viridis',
            vmin=phi_min,
            vmax=phi_max,
            aspect='auto'
        )

        ax.set_xlabel("phi value")
        ax.set_ylabel("y-axis (along x=1)")
        ax.set_title(f"Phi profile at iteration {iteration}")
        ax.margins(x=0, y=0)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(phi_min, phi_max, 3))

        filename = f"{plot_name}_{iteration:0{self.FILENAME_PADDING_WIDTH}d}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)        

        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


    # 2D Velocity map
    def velocity_map(self, ax, u_magnitude, iteration, title):
        """
        Save a velocity map as a standalone PNG without popping up, 
        while optionally updating a provided interactive axis.
        """

        # -------------------------
        # 1. Standalone figure (silent)
        # -------------------------
        fig_standalone = plt.figure(figsize=(8, 5))
        ax_standalone = fig_standalone.add_subplot(111)
        im = ax_standalone.imshow(u_magnitude.T, cmap='viridis', origin='lower')
        ax_standalone.set_xlabel('x')
        ax_standalone.set_ylabel('y')
        ax_standalone.set_title(title)
        plt.colorbar(im, ax=ax_standalone, label='Velocity')

        # Save PNG silently
        #script_dir = os.path.dirname(os.path.abspath(__file__))
        script_dir = self.script_dir
        images_dir = os.path.join(script_dir, "FreesurfaceImages")
        os.makedirs(images_dir, exist_ok=True)

        iteration_str = f"{max(iteration, self.TOTAL_ITERATIONS):0{self.FILENAME_PADDING_WIDTH}d}"
        simplified_title = title.replace('Velocity [u$_x$] map', 'velocity_ux').replace('Velocity [u$_y$] map', 'velocity_uy')
        filename = f"{simplified_title}_{iteration_str}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)

        fig_standalone.savefig(save_path, dpi=300, bbox_inches='tight')
        self.debug_log('INIT', 'Saved velocity map: %s', save_path)

        # Close standalone figure to suppress display
        plt.close(fig_standalone)

        # -------------------------
        # 2. Optional live plot on provided ax
        # -------------------------
        if ax is not None:
            im_live = ax.imshow(u_magnitude.T, cmap='viridis', origin='lower')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(title)
            plt.colorbar(im_live, ax=ax, label='Velocity')
            # Do NOT call plt.show() here; live figure updated elsewhere


    def chemical_potential_map(self, ax, mu_phi, iteration, title, label="Chemical Potential μϕ"):
        """
        Save a chemical potential map as a standalone PNG without popping up,
        while optionally updating a provided interactive axis.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes or None
            Optional axis to update live.
        mu_phi : 2D array
            Chemical potential field (e.g., output of chemical_potential_Zhang).
        iteration : int
            Current iteration number for filename.
        title : str
            Title for the figure and colorbar.
        """

        # -------------------------
        # 1. Standalone figure (silent)
        # -------------------------
        fig_standalone = plt.figure(figsize=(8, 5))
        ax_standalone = fig_standalone.add_subplot(111)
        
        # Clip colormap to 1st–99th percentile so interface detail isn't crushed by bulk extremes
        vmin, vmax = np.nanpercentile(mu_phi, [1, 99])
        if vmin == vmax:
            vmin, vmax = None, None
        im = ax_standalone.imshow(mu_phi.T, cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax)
        ax_standalone.set_xlabel('x')
        ax_standalone.set_ylabel('y')
        ax_standalone.set_title(label)
        plt.colorbar(im, ax=ax_standalone, label=label)

        # Save PNG silently
        script_dir = self.script_dir
        images_dir = os.path.join(script_dir, "FreesurfaceImages")
        os.makedirs(images_dir, exist_ok=True)

        iteration_str = f"{iteration:0{self.FILENAME_PADDING_WIDTH}d}"
        filename = f"{title}_{iteration_str}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)
        fig_standalone.savefig(save_path, dpi=300, bbox_inches='tight')
        self.debug_log('INIT', 'Saved velocity map: %s', save_path)        

        # Close standalone figure to suppress display
        plt.close(fig_standalone)

        # -------------------------
        # 2. Optional live plot on provided ax
        # -------------------------
        if ax is not None:
            im_live = ax.imshow(mu_phi.T, cmap='coolwarm', origin='lower')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(title)
            plt.colorbar(im_live, ax=ax, label=label)            


    def phi_x_axis_plot_1(self, ax, phi_center, yc, iteration, title, label1):
        """
        Plot _phi along the centerline y = (Yn+2)//2 versus x.
        Positive phi appears above the x-axis, negative below.
        Saves a standalone PNG and optionally updates a provided axis.
        """

        # Compute centerline index
        #yc = (Yn + 2) // 2

        # Extract centerline profile (assuming phi is [x, y] or [Nx, Ny])
        #phi_center = phi[:, yc]

        # -------------------------
        # 1. Standalone figure (silent)
        # -------------------------
        fig_standalone = plt.figure(figsize=(8, 4))
        ax_standalone = fig_standalone.add_subplot(111)

        x = np.arange(len(phi_center))
        ax_standalone.plot(x, phi_center, linewidth=1.5)

        ax_standalone.axhline(0.0, color='k', linewidth=0.8)  # x-axis reference
        ax_standalone.set_xlabel('x')
        ax_standalone.set_ylabel(r'$\phi$')
        ax_standalone.set_title(f"{title} (centerline y={yc})")
        ax_standalone.grid(True, linestyle='--', alpha=0.4)

        # Save PNG silently
        #script_dir = os.path.dirname(os.path.abspath(__file__))
        script_dir = self.script_dir
        images_dir = os.path.join(script_dir, "FreesurfaceImages")
        os.makedirs(images_dir, exist_ok=True)

        iteration_str = f"{max(iteration, self.TOTAL_ITERATIONS):0{self.FILENAME_PADDING_WIDTH}d}"
        filename = f"{title}_{iteration_str}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)
        fig_standalone.savefig(save_path, dpi=300, bbox_inches='tight')
        self.debug_log('INIT', 'Saved phi centerline plot: %s', save_path)

        plt.close(fig_standalone)

        # -------------------------
        # 2. Optional live plot on provided ax
        # -------------------------
        if ax is not None:
            ax.clear()
            ax.plot(x, phi_center, linewidth=1.5)
            ax.axhline(0.0, color='k', linewidth=0.8)
            ax.set_xlabel('x')
            ax.set_ylabel(label1)
            ax.set_title(f"{title} (centerline y={yc})")
            ax.grid(True, linestyle='--', alpha=0.4)          


    def plot_left_wall_all_nodes(self, iteration, node_data):
        """
        node_data = list of tuples:
        [(offset, [phi0..phi6], diff0), ...]
        """

        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, 6))

        x_nodes = np.arange(7)  # 0..6

        for offset, values, diff0 in node_data:
            ax.plot(
                x_nodes,
                values,
                marker='o',
                linewidth=1.5,
                label=f"{offset}"
            )

            # annotate diff at first point
            ax.annotate(f"{diff0:.2e}",
                        (x_nodes[0], values[0]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8)

        ax.set_xlabel("Node index (0..6)")
        ax.set_ylabel(r"$\phi$")
        ax.set_title(f"Left wall phi (iteration {iteration})")

        # This is key: shows [4..-4] as labels
        ax.legend(title="Offset", loc="best")

        ax.grid(True, linestyle='--', alpha=0.4)

        # save
        iteration_str = str(iteration)
        filename = f"left_wall_nodes_{iteration_str}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)

        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        self.debug_log('INIT', 'Saved left wall ALL nodes plot: %s', save_path)              


    def phi_x_axis_plot_2(self, ax, phi_center, phi_w_center, yc, iteration, title, label1, label2):
        """
        Plot _phi and _phi_w along the centerline y = yc versus x.
        Positive phi appears above the x-axis, negative below.
        Saves a standalone PNG and optionally updates a provided axis.
        """

        # -------------------------
        # 1. Standalone figure (silent)
        # -------------------------
        fig_standalone = plt.figure(figsize=(8, 4))
        ax_standalone = fig_standalone.add_subplot(111)

        x = np.arange(len(phi_center))
        ax_standalone.plot(x, phi_center, label=label1, linewidth=1.5, color='b')
        ax_standalone.plot(
            x, phi_w_center,
            label=label2,
            linewidth=1.5,
            color='r',
            linestyle='--'
        )

        ax_standalone.axhline(0.0, color='k', linewidth=0.8)  # x-axis reference
        ax_standalone.set_xlabel('x')
        ax_standalone.set_ylabel(label1)
        ax_standalone.set_title(f"{title} (centerline y={yc})")
        ax_standalone.grid(True, linestyle='--', alpha=0.4)
        ax_standalone.legend()

        # Save PNG silently
        #script_dir = os.path.dirname(os.path.abspath(__file__))
        script_dir = self.script_dir
        images_dir = os.path.join(script_dir, "FreesurfaceImages")
        os.makedirs(images_dir, exist_ok=True)

        iteration_str = f"{iteration:0{self.FILENAME_PADDING_WIDTH}d}"
        filename = f"{title}_{iteration_str}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)
        fig_standalone.savefig(save_path, dpi=300, bbox_inches='tight')
        self.debug_log('INIT', 'Saved ext. phi centerline plot: %s', save_path)

        plt.close(fig_standalone)

        # -------------------------
        # 2. Optional live plot on provided ax
        # -------------------------
        if ax is not None:
            ax.clear()
            ax.plot(x, phi_center, label=r'$\phi$', linewidth=1.5, color='b')
            ax.plot(x, phi_w_center, label=r'$\phi_w$', linewidth=1.5, color='r', linestyle='--')
            ax.axhline(0.0, color='k', linewidth=0.8)
            ax.set_xlabel('x')
            ax.set_ylabel(r'$\phi$')
            ax.set_title(f"{title} (centerline y={yc})")
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend()


    def phi_x_axis_plot_3(
            self, ax,
            series1, series2, series3,
            label1, label2, label3,
            axis1='left', axis2='right', axis3='right',
            yc=None, iteration=None, title="", xlabel='x'
        ):
        """
        Plot three arrays along centerline y = yc versus x.

        Each series can be plotted on the left or right y-axis.

        axis1, axis2, axis3:
            'left' / 'right' or 0 / 1
        """

        def is_right(axis):
            """Return True if the series should be plotted on the right y-axis."""
            return str(axis).lower() in ('right', '1')

        x = np.arange(len(series1))

        # -------------------------
        # 1. Standalone figure
        # -------------------------
        fig = plt.figure(figsize=(8, 4))
        ax_left = fig.add_subplot(111)
        ax_right = ax_left.twinx()

        lines = []

        # --- Plot each series on the chosen axis ---
        if is_right(axis1):
            lines += ax_right.plot(x, series1, label=label1, linewidth=1.5, color='b')
        else:
            lines += ax_left.plot(x, series1, label=label1, linewidth=1.5, color='b')

        if is_right(axis2):
            lines += ax_right.plot(x, series2, label=label2, linewidth=1.5, color='r', linestyle='--')
        else:
            lines += ax_left.plot(x, series2, label=label2, linewidth=1.5, color='r', linestyle='--')

        if is_right(axis3):
            lines += ax_right.plot(x, series3, label=label3, linewidth=1.5, color='g', linestyle=':')
        else:
            lines += ax_left.plot(x, series3, label=label3, linewidth=1.5, color='g', linestyle=':')

        # --- Axes styling ---
        ax_left.axhline(0.0, color='k', linewidth=0.8)
        ax_left.set_xlabel(xlabel)

        # Only join labels that are actually on the axis
        ax_left.set_ylabel(
            ", ".join(str(l) for l, a in zip([label1, label2, label3], [axis1, axis2, axis3]) if not is_right(a))
        )
        ax_right.set_ylabel(
            ", ".join(str(l) for l, a in zip([label1, label2, label3], [axis1, axis2, axis3]) if is_right(a))
        )

        ax_left.set_title(f"{title} (centerline y={yc})" if yc is not None else title)
        ax_left.grid(True, linestyle='--', alpha=0.4)

        # Combined legend
        labels = [l.get_label() for l in lines]
        ax_left.legend(lines, labels)

        # --- Save PNG ---
        if iteration is not None:
            #script_dir = os.path.dirname(os.path.abspath(__file__))
            script_dir = self.script_dir
            images_dir = os.path.join(script_dir, "FreesurfaceImages")
            os.makedirs(images_dir, exist_ok=True)

            iteration_str = f"{iteration:0{self.FILENAME_PADDING_WIDTH}d}"
            filename = f"{title}_{iteration_str}.png"
            save_path = os.path.join(self.IMAGES_SUBDIR, filename)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.debug_log('INIT', 'Saved ext. phi centerline plot: %s', save_path)

        plt.close(fig)

        # -------------------------
        # 2. Optional live plot
        # -------------------------
        if ax is not None:
            ax.clear()
            ax_left = ax
            ax_right = ax_left.twinx()
            lines = []

            if is_right(axis1):
                lines += ax_right.plot(x, series1, label=label1, linewidth=1.5, color='b')
            else:
                lines += ax_left.plot(x, series1, label=label1, linewidth=1.5, color='b')

            if is_right(axis2):
                lines += ax_right.plot(x, series2, label=label2, linewidth=1.5, color='r', linestyle='--')
            else:
                lines += ax_left.plot(x, series2, label=label2, linewidth=1.5, color='r', linestyle='--')

            if is_right(axis3):
                lines += ax_right.plot(x, series3, label=label3, linewidth=1.5, color='g', linestyle=':')
            else:
                lines += ax_left.plot(x, series3, label=label3, linewidth=1.5, color='g', linestyle=':')

            ax_left.axhline(0.0, color='k', linewidth=0.8)
            ax_left.set_xlabel(xlabel)
            ax_left.set_title(f"{title} (centerline y={yc})" if yc is not None else title)
            ax_left.grid(True, linestyle='--', alpha=0.4)

            ax_left.set_ylabel(
                ", ".join(str(l) for l, a in zip([label1, label2, label3], [axis1, axis2, axis3]) if not is_right(a))
            )
            ax_right.set_ylabel(
                ", ".join(str(l) for l, a in zip([label1, label2, label3], [axis1, axis2, axis3]) if is_right(a))
            )

            # Combined legend
            labels = [l.get_label() for l in lines]
            ax_left.legend(lines, labels)


    def phi_x_axis_plot_4(
            self, ax,
            series1, series2, series3, series4,
            label1, label2, label3, label4,
            axis1='left', axis2='left', axis3='right', axis4='right',
            yc=None, 
            iteration=None, 
            title=""
        ):
        """
        Plot 4 arrays along centerline y = yc versus x.

        Each series can be plotted on the left or right y-axis.

        axis1, axis2, axis3:
            'left' / 'right' or 0 / 1
        """

        print(f"ENTER phi_x_axis_plot_4 | ax={ax} | title='{title}' | iter={iteration} | time={id(self):x}")        

        def is_right(axis):
            """Return True if the series should be plotted on the right y-axis."""
            return str(axis).lower() in ('right', '1')

        x = np.arange(len(series1))

        # -------------------------
        # 1. Standalone figure
        # -------------------------
        fig = plt.figure(figsize=(8, 4))
        ax_left = fig.add_subplot(111)
        ax_right = ax_left.twinx()
        print(len(fig.axes))          # should be 2 (left + right)

        print("Creating new figure — number of axes so far:", len(fig.axes))
        print("Plotting series — lengths:", len(series1), len(series2), len(series3), len(series4))

        lines = []

        # --- Plot each series on the chosen axis ---
        if is_right(axis1):
            lines += ax_right.plot(x, series1, label=label1, linewidth=1.5, color='b')
        else:
            lines += ax_left.plot(x, series1, label=label1, linewidth=1.5, color='b')

        if is_right(axis2):
            lines += ax_right.plot(x, series2, label=label2, linewidth=1.5, color='r', linestyle='--')
        else:
            lines += ax_left.plot(x, series2, label=label2, linewidth=1.5, color='r', linestyle='--')

        if is_right(axis3):
            lines += ax_right.plot(x, series3, label=label3, linewidth=1.5, color='g', linestyle=':')
        else:
            lines += ax_left.plot(x, series3, label=label3, linewidth=1.5, color='g', linestyle=':')

        if is_right(axis4):
            lines += ax_right.plot(x, series4, label=label4, linewidth=1.5, color='g', linestyle=':')
        else:
            lines += ax_left.plot(x, series4, label=label4, linewidth=1.5, color='g', linestyle=':')            

        print("Lines plotted so far:", len(lines))            

        # --- Axes styling ---
        ax_left.axhline(0.0, color='k', linewidth=0.8)
        ax_left.set_xlabel('x')

        # Only join labels that are actually on the axis
        ax_left.set_ylabel(
            ", ".join(str(l) for l, a in zip([label1, label2, label3, label4], [axis1, axis2, axis3, axis4]) if not is_right(a))
        )
        ax_right.set_ylabel(
            ", ".join(str(l) for l, a in zip([label1, label2, label3, label4], [axis1, axis2, axis3, axis4]) if is_right(a))
        )

        ax_left.set_title(f"{title} (centerline y={yc})" if yc is not None else title)
        ax_left.grid(True, linestyle='--', alpha=0.4)

        # Combined legend
        labels = [l.get_label() for l in lines]
        ax_left.legend(lines, labels)

        # --- Save PNG ---
        if iteration is not None:
            #script_dir = os.path.dirname(os.path.abspath(__file__))
            script_dir = self.script_dir
            images_dir = os.path.join(script_dir, "FreesurfaceImages")
            os.makedirs(images_dir, exist_ok=True)

            iteration_str = f"{iteration:0{self.FILENAME_PADDING_WIDTH}d}"
            filename = f"{title}_{iteration_str}.png"
            save_path = os.path.join(self.IMAGES_SUBDIR, filename)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.debug_log('INIT', 'Saved ext. phi centerline plot: %s', save_path)

        plt.close(fig)


    def wall_force_quiver(
            self,
            fc,
            _phi,           # full 2D phi field  (Xn+2, Yn+2)
            net_force,      # full 2D force field (2, Xn+2, Yn+2): net_force[0]=Fx, net_force[1]=Fy
            iteration,
            yc=None,
            strip_width=40,  # nodes to show from each wall (interior side)
            stride=3,        # plot every Nth vector to avoid clutter
        ):
        """
        Two-panel quiver plot of the net force field near each wall.

        Left panel  : x = 1 .. strip_width   (left wall region)
        Right panel : x = Xn-strip_width .. Xn  (right wall region)

        Background: phi field coloured by phase (same RdBu_r scheme).
        Vectors:    net_force direction and magnitude.
                    Arrow colour follows phi so you see liquid/interface/gas phase.
        Interface contour drawn in white.
        """
        Xn2, Yn2 = _phi.shape          # Xn+2, Yn+2
        Xn = Xn2 - 2

        phi_G  = fc.phi_star_G
        phi_L  = fc.phi_star_L
        phi_mid = 0.5 * (phi_G + phi_L)

        # ─── Strips: interior nodes only (no ghost) ───────────────────────
        # Left strip: x-indices 1 .. strip_width
        x_l  = np.arange(1, strip_width + 1)
        phi_l  = _phi[x_l, :]          # (strip_width, Yn+2)
        fx_l   = net_force[0][x_l, :]
        fy_l   = net_force[1][x_l, :]

        # Right strip: x-indices Xn-strip_width+1 .. Xn
        x_r  = np.arange(Xn - strip_width + 1, Xn + 1)
        phi_r  = _phi[x_r, :]
        fx_r   = net_force[0][x_r, :]
        fy_r   = net_force[1][x_r, :]

        # ─── Quiver grid (strided) ────────────────────────────────────────
        # Meshgrid for imshow: x-axis = x-index, y-axis = y-index
        y_all  = np.arange(Yn2)

        def make_quiver_data(phi_strip, fx_strip, fy_strip, x_base):
            xs = x_base[::stride]
            ys = y_all[::stride]
            XI, YI = np.meshgrid(xs, ys, indexing='ij')
            FX = fx_strip[::stride, ::stride]
            FY = fy_strip[::stride, ::stride]
            PHI_q = phi_strip[::stride, ::stride]
            return XI, YI, FX, FY, PHI_q

        XI_l, YI_l, FX_l, FY_l, PHI_ql = make_quiver_data(phi_l, fx_l, fy_l, x_l)
        XI_r, YI_r, FX_r, FY_r, PHI_qr = make_quiver_data(phi_r, fx_r, fy_r, x_r)

        # ─── Figure ───────────────────────────────────────────────────────
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

        phi_kw = dict(origin='lower', aspect='auto',
                      cmap='RdBu_r', vmin=phi_G, vmax=phi_L)

        for ax, phi_strip, XI, YI, FX, FY, PHI_q, label in [
            (ax_l, phi_l, XI_l, YI_l, FX_l, FY_l, PHI_ql, "Left wall"),
            (ax_r, phi_r, XI_r, YI_r, FX_r, FY_r, PHI_qr, "Right wall"),
        ]:
            # phi background
            ax.imshow(phi_strip.T, extent=[phi_strip.shape[0], 0, 0, Yn2],
                      **phi_kw)
            # interface contour
            ax.contour(phi_strip.T, levels=[phi_mid], colors='white',
                       linewidths=1.2, origin='upper')

            # magnitude for colour and scaling
            mag = np.hypot(FX, FY)
            mag_max = mag.max()
            if mag_max == 0:
                mag_max = 1.0

            # normalise arrows to unit length; scale by mag relative to max
            U = FX / (mag_max + 1e-30)
            V = FY / (mag_max + 1e-30)

            # colour arrows by phi value (same cmap)
            phi_flat = PHI_q.ravel()
            norm = plt.Normalize(vmin=phi_G, vmax=phi_L)
            colors = plt.cm.RdBu_r(norm(phi_flat))

            # quiver: positions in strip-local x, global y
            # ravel for scatter-style quiver
            xi_flat  = XI.ravel()
            yi_flat  = YI.ravel()
            u_flat   = U.ravel()
            v_flat   = V.ravel()
            mag_flat = mag.ravel() / mag_max

            q = ax.quiver(xi_flat, yi_flat, u_flat, v_flat,
                          color=colors,
                          scale=strip_width / stride,   # tune: larger = shorter arrows
                          scale_units='width',
                          width=0.004,
                          alpha=0.85)

            if yc is not None:
                ax.axhline(yc, color='lime', lw=1.0, ls='--', alpha=0.8)

            ax.set_title(label, fontsize=12)
            ax.set_xlabel("x-index", fontsize=10)

        ax_l.set_ylabel("y-index  (0 = bottom/liquid)", fontsize=10)

        # ─── Colourbar (phi) ─────────────────────────────────────────────
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=phi_G, vmax=phi_L))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax_l, ax_r], shrink=0.7, pad=0.02)
        cbar.set_label("phi  (red = liquid, blue = gas)", fontsize=10)

        theta_str = f"  θ={fc.vf_theta:.0f}°" if hasattr(fc, 'vf_theta') else ""
        iter_str  = f"  iter={iteration}"
        fig.suptitle(f"Net force vectors near walls{theta_str}{iter_str}", fontsize=13)
        plt.tight_layout()

        # ─── Save ─────────────────────────────────────────────────────────
        iteration_str = f"{iteration:0{self.FILENAME_PADDING_WIDTH}d}"
        filename  = f"wall_force_quiver_{iteration_str}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)


    def phi_x_axis_plot_5(self, ax, series1, series2, series3, series4, series5,
                        yc, iteration, title,
                        label1, label2, label3, label4, label5):
        """
        Plot five arrays along the centerline y = yc versus x.
        Positive phi appears above the x-axis, negative below.
        Saves a standalone PNG and optionally updates a provided axis.
        """

        # -------------------------
        # 1. Standalone figure (silent)
        # -------------------------
        fig_standalone = plt.figure(figsize=(8, 4))
        ax_standalone = fig_standalone.add_subplot(111)

        # Determine global x-range based on all series
        max_len = max(len(series1), len(series2), len(series3), len(series4), len(series5))
        x = np.arange(max_len)

        # List of series and labels for easy looping
        series_list = [series1, series2, series3, series4, series5]
        labels = [label1, label2, label3, label4, label5]
        colors = ['b', 'r', 'g', 'c', 'm']           
        linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1, 1, 1))]  

        for s, lbl, col, ls in zip(series_list, labels, colors, linestyles):
            ax_standalone.plot(np.arange(len(s)), s, label=lbl, linewidth=1.5, color=col, linestyle=ls)

        ax_standalone.axhline(0.0, color='k', linewidth=0.8)
        ax_standalone.set_xlabel('x')
        ax_standalone.set_ylabel(label1)
        ax_standalone.set_title(f"{title} (centerline y={yc})")
        ax_standalone.grid(True, linestyle='--', alpha=0.4)
        ax_standalone.legend()
        
        # Set x-limits to the global range
        ax_standalone.set_xlim(0, max_len - 1)

        # Save PNG silently
        #script_dir = os.path.dirname(os.path.abspath(__file__))
        script_dir = self.script_dir
        images_dir = os.path.join(script_dir, "FreesurfaceImages")
        os.makedirs(images_dir, exist_ok=True)

        iteration_str = f"{iteration:0{self.FILENAME_PADDING_WIDTH}d}"
        filename = f"{title}_{iteration_str}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)
        fig_standalone.savefig(save_path, dpi=300, bbox_inches='tight')
        self.debug_log('INIT', 'Saved ext. phi centerline plot: %s', save_path)
        plt.close(fig_standalone)

        # -------------------------
        # 2. Optional live plot on provided ax
        # -------------------------
        if ax is not None:
            ax.clear()
            for s, lbl, col, ls in zip(series_list, labels, colors, linestyles):
                ax.plot(np.arange(len(s)), s, label=lbl, linewidth=1.5, color=col, linestyle=ls)

            ax.axhline(0.0, color='k', linewidth=0.8)
            ax.set_xlabel('x')
            ax.set_ylabel(label1)
            ax.set_title(f"{title} (centerline y={yc})")
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend()
            
            # Set x-limits to the same global range
            ax.set_xlim(0, max_len - 1)


    def filter_u_ckl_fullrange(self, velocities_dict, iterationsOfInterest):
        filtered_velocities = {iter_: velocities_dict[iter_] for iter_ in iterationsOfInterest if iter_ in velocities_dict}

        return filtered_velocities   


    def plot_bounds_ext(self, results, context, ax=None, series_labels=None, k=None, script_filename=None):
        if results is None or not isinstance(results, (list, tuple)) or not results or len(results[0]) < 2:
            self.debug_log("results must be non-empty list of ({%s}, values)", results)
            return

        if isinstance(results, np.ndarray):
            results = results.tolist()

        # Detect if data is numeric
        sample_values = results[0][1:]
        is_numeric = all(isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_))
                        for v in sample_values)

        # --- Convert to relative values ONLY if numeric ---
        if is_numeric:
            phi0 = sample_values
            results_proc = [
                (r[0], *[val - base for val, base in zip(r[1:], phi0)])
                for r in results
            ]
            ylabel = f"{context} (Δ from initial)"
            title = f"{context} deviation vs Iteration"
        else:
            results_proc = results
            ylabel = context
            title = f"{context} vs Iteration"

        iterations = [r[0] for r in results_proc]
        data_series = list(zip(*[r[1:] for r in results_proc]))
        n_series = len(data_series)

        series_labels = series_labels or [f"{context}_{i+1}" for i in range(n_series)]

        # --- Standalone figure ---
        fig_standalone = plt.figure(figsize=(8, 5))
        for series, label in zip(data_series, series_labels):
            plt.plot(iterations, series, label=label)

        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(
            ncol=2,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            fontsize='small',
            frameon=True
        )
        plt.grid(True)

        if script_filename is None:
            script_filename = self.SCRIPT_FILENAME

        filename = f"{context.replace(' ', '_')}_{self.TOTAL_ITERATIONS:0{self.FILENAME_PADDING_WIDTH}d}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig_standalone)

        self.debug_log('INIT', 'Saved plot_bounds_ext: %s', save_path)

        # --- Plot on provided ax ---
        if ax is not None:
            plt.sca(ax)
            for series, label in zip(data_series, series_labels):
                plt.plot(iterations, series, label=label)

            plt.xlabel("Iteration")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend(
                ncol=2,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                fontsize='small',
                frameon=True
            )
            plt.grid(True)


    def plot_momentum_bounds(self, results, _filename, ax=None):
        iterations = [r[0] for r in results]
        invariant = [r[3] for r in results]
        
        # Create standalone figure
        fig_standalone = plt.figure(figsize=(8, 5))
        plt.plot(iterations, invariant, label="total_mom_x", color="green")
        plt.axhline(0, color="black", linestyle="--", alpha=0.5)
        plt.xlabel("Iteration")
        plt.ylabel("Total invariant")
        plt.title("Total invariant conservation")
        plt.legend()
        plt.grid(True)
        
        # Save standalone plot
        filename = f"{_filename}_{self.TOTAL_ITERATIONS:0{self.FILENAME_PADDING_WIDTH}d}.png"

        save_path = os.path.join(self.IMAGES_SUBDIR, filename)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig_standalone)
        self.debug_log('INIT', 'Saved plot_momentum_bounds: %s', save_path)
        
        # Plot on provided ax
        if ax is not None:
            plt.sca(ax)
            plt.plot(iterations, invariant, label="total_mom_x", color="green")
            plt.axhline(0, color="black", linestyle="--", alpha=0.5)
            plt.xlabel("Iteration")
            plt.ylabel("Total invariant")
            plt.title("Total invariant conservation")
            plt.legend()
            plt.grid(True)


    def save_phi_snapshot(self, _phi, iteration, phi_star_G, phi_star_L):
        """
        Save a snapshot plot of the order parameter phi and print min/max/mean stats.

        Args:
            _phi: 2D numpy array of phi values (Ny, Nx).
            iteration: Current iteration number (int).
            phi_star_G: Lower bound for colorbar (float).
            phi_star_L: Upper bound for colorbar (float).
            script_dir: Directory to save PNG (str; default: script's directory).

        Returns:
            None (saves PNG and prints stats).
        """
        # Create plot
        fig = Figure(figsize=(8,6))
        canvas = FigureCanvas(fig) 
        ax = fig.add_subplot(111)
        #plt.figure(figsize=(8, 6))

        im = ax.imshow(_phi.T, origin='lower', cmap='RdBu', vmin=phi_star_G, vmax=phi_star_L)

        # Create a divider so the colorbar matches the axes height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)   # width & spacing
        cbar = fig.colorbar(im, cax=cax, label='phi')
        # -------------------------

        ax.set_title(f'Order parameter phi at iteration {iteration}')
        ax.set_xlabel('x-index')
        ax.set_ylabel('y-index')

        # Save PNG 
        _filename = f'phi_snapshot_iter_{iteration:0{self.FILENAME_PADDING_WIDTH}d}'
        filename = f"{_filename}.png"

        save_path = os.path.join(self.IMAGES_SUBDIR, filename)

        canvas.print_figure(save_path, dpi=150, bbox_inches='tight')

        # Print stats
        phi_min = np.min(_phi)
        phi_max = np.max(_phi)
        phi_mean = np.mean(_phi)
        self.debug_log('ITER', 'phi at iter %d: min=%.3f, max=%.3f, mean=%.3f', 
                iteration, phi_min, phi_max, phi_mean)
        self.debug_log('INIT', 'Saved phi snapshot: %s', save_path)


    def plot_capillary_forces(
            self,
            forces,   # shape (2, Xn, Yn)
            yc=None,
            iteration=None,
            title=None,
            figsize=(7, 9),
            scale=None
        ):
        """
        Plot capillary force vectors in the full 2D domain.
        Only non-zero vectors are plotted.

        forces: (2, Xn, Yn)
            forces[0, x, y] = Fx
            forces[1, x, y] = Fy
        """

        import numpy as np
        import matplotlib.pyplot as plt
        import os

        _, Xn, Yn = forces.shape

        # Extract components
        Fx = forces[0, :, :]
        Fy = forces[1, :, :]

        # --- FIX ORIENTATION ---
        # Convert from (x, y) → (y, x) for plotting
        Fx = Fx.T   # now shape (Yn, Xn)
        Fy = Fy.T

        Ny, Nx = Fx.shape

        # Grid
        X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))

        # Mask non-zero vectors
        eps = 1e-14
        mask = (np.abs(Fx) > eps) | (np.abs(Fy) > eps)

        # Prepare plot
        fig, ax = plt.subplots(figsize=figsize)

        if np.any(mask):
            X_nz = X[mask]
            Y_nz = Y[mask]
            Fx_nz = Fx[mask]
            Fy_nz = Fy[mask]

            # Normalize (direction only)
            mag = np.sqrt(Fx_nz**2 + Fy_nz**2)
            Fx_nz /= mag
            Fy_nz /= mag

            if scale is None:
                scale = 1.0

            ax.quiver(X_nz, Y_nz, Fx_nz, Fy_nz,
                    angles='xy', scale_units='xy', scale=scale)

        else:
            print("All force vectors are zero → empty plot")

        # Axes
        ax.set_xlim(0, Nx - 1)
        ax.set_ylim(0, Ny - 1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if title is None:
            title = "Capillary force vectors"
        if yc is not None:
            title += f"  (yc={yc})"

        ax.set_title(title)
        ax.grid(True, ls='--', alpha=0.3)

        # Save
        if iteration is not None:
            script_filename = self.SCRIPT_FILENAME
            filename = f"{title.replace(' ', '_')}_{iteration:0{self.FILENAME_PADDING_WIDTH}d}.png"
            save_path = os.path.join(self.IMAGES_SUBDIR, filename)
            os.makedirs(self.IMAGES_SUBDIR, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            self.debug_log('INIT', 'Saved capillary force plot: %s', save_path)

        plt.close(fig)


    def phi_boundary_forces_vertical(
            self,
            series1, series2, series3=None, series4=None,
            label_left_x="Fx", label_left_y="Fy",
            label_right_x=None, label_right_y=None,
            yc=None, iteration=None, title="Boundary forces", figsize=(7, 5)
        ):
        """
        Plot 2 or 4 vertical (y-profile) force components at a wall boundary.
        With 4 series: 2x2 grid (series1/2 top row, series3/4 bottom row).
        """
        import matplotlib.pyplot as plt
        import os

        has_four = series3 is not None and series4 is not None
        if has_four:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            pairs = [
                (axes[0, 0], series1, label_left_x),
                (axes[0, 1], series2, label_left_y),
                (axes[1, 0], series3, label_right_x or "series3"),
                (axes[1, 1], series4, label_right_y or "series4"),
            ]
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            pairs = [
                (axes[0], series1, label_left_x),
                (axes[1], series2, label_left_y),
            ]

        fig.suptitle(f"{title}  iter={iteration}", fontsize=10)

        for ax, data, label in pairs:
            y = range(len(data))
            ax.plot(data, y)
            ax.set_ylabel("y node")
            ax.set_title(label, fontsize=9)
            ax.grid(True, ls='--', alpha=0.3)
            if yc is not None:
                ax.axhline(yc, color='red', ls='--', lw=0.8, label=f"yc={yc}")
                ax.legend(fontsize=7)

        fig.tight_layout()

        if iteration is not None:
            filename = f"{title.replace(' ', '_')}_{iteration:0{self.FILENAME_PADDING_WIDTH}d}.png"
            save_path = os.path.join(self.IMAGES_SUBDIR, filename)
            os.makedirs(self.IMAGES_SUBDIR, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            self.debug_log('INIT', 'Saved boundary force plot: %s', save_path)

        plt.close(fig)


    def plot_fsv_diagnostic(self, fc, _phi, fsv_field, iteration):
        """
        Three-panel diagnostic for the m-vector surface tension force.
        Panel 1: phi field (phase map, interface contour).
        Panel 2: Fsv_x — horizontal force component (should be near-zero in bulk).
        Panel 3: Fsv_y — vertical force component (drives capillary rise/depression).
        Force is non-zero only in the wall strip where csf_m_fc applies it.
        """
        fsv_x = fsv_field[0]
        fsv_y = fsv_field[1]
        phi_mid = (fc.phi_star_G + fc.phi_star_L) / 2

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # --- Panel 1: phi ---
        im0 = axes[0].imshow(_phi.T, origin='lower', cmap='RdBu_r',
                             vmin=fc.phi_star_G, vmax=fc.phi_star_L)
        axes[0].contour(_phi.T, levels=[phi_mid], colors='white', linewidths=1)
        plt.colorbar(im0, ax=axes[0])
        axes[0].set_title(f'phi  θ={fc.vf_theta:.0f}°')

        # --- Panel 2: Fsv_x ---
        xlim = max(np.max(np.abs(fsv_x)), 1e-10)
        im1 = axes[1].imshow(fsv_x.T, origin='lower', cmap='coolwarm',
                             vmin=-xlim, vmax=xlim)
        axes[1].contour(_phi.T, levels=[phi_mid], colors='black', linewidths=1)
        plt.colorbar(im1, ax=axes[1])
        axes[1].set_title(f'Fsv_x (horizontal)  θ={fc.vf_theta:.0f}°')

        # --- Panel 3: Fsv_y ---
        ylim = max(np.max(np.abs(fsv_y)), 1e-10)
        im2 = axes[2].imshow(fsv_y.T, origin='lower', cmap='coolwarm',
                             vmin=-ylim, vmax=ylim)
        axes[2].contour(_phi.T, levels=[phi_mid], colors='black', linewidths=1)
        plt.colorbar(im2, ax=axes[2])
        axes[2].set_title(f'Fsv_y (vertical)  θ={fc.vf_theta:.0f}°')

        for ax in axes:
            ax.set_xlabel('x-index')
            ax.set_ylabel('y-index')

        theta_str = f'{fc.vf_theta:.0f}'
        plt.suptitle(f'Fsv diagnostic (m-vector)  iter={iteration}  θ={theta_str}°')
        plt.tight_layout()
        fname = os.path.join(self.IMAGES_SUBDIR,
                             f'fsv_diagnostic_theta{theta_str}_iter{iteration:05d}.png')
        plt.savefig(fname, dpi=120)
        plt.close()

    def plot_chi_diagnostic(self, fc, _phi, _rho, chi_field, fsv_field, iteration):
        """
        Three-panel diagnostic for chi-based Fsv (original Inamuro formulation).
        Panel 1: phi field.
        Panel 2: chi (mean curvature of density isosurface).
        Panel 3: Fsv_y (y-component of chi-based surface tension force).
        """
        fsv_y   = fsv_field[1]
        phi_mid = (fc.phi_star_G + fc.phi_star_L) / 2

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # --- Panel 1: phi ---
        im0 = axes[0].imshow(_phi.T, origin='lower', cmap='RdBu_r',
                             vmin=fc.phi_star_G, vmax=fc.phi_star_L)
        axes[0].contour(_phi.T, levels=[phi_mid], colors='white', linewidths=1)
        plt.colorbar(im0, ax=axes[0])
        axes[0].set_title(f'phi  θ={fc.vf_theta:.0f}°')

        # --- Panel 2: chi ---
        clim = max(np.max(np.abs(chi_field)), 1e-10)
        im1 = axes[1].imshow(chi_field.T, origin='lower', cmap='coolwarm',
                             vmin=-clim, vmax=clim)
        axes[1].contour(_phi.T, levels=[phi_mid], colors='black', linewidths=1)
        plt.colorbar(im1, ax=axes[1])
        axes[1].set_title(f'chi (curvature)  θ={fc.vf_theta:.0f}°')

        # --- Panel 3: Fsv_y (chi-based) ---
        flim = max(np.max(np.abs(fsv_y)), 1e-10)
        im2 = axes[2].imshow(fsv_y.T, origin='lower', cmap='coolwarm',
                             vmin=-flim, vmax=flim)
        axes[2].contour(_phi.T, levels=[phi_mid], colors='black', linewidths=1)
        plt.colorbar(im2, ax=axes[2])
        axes[2].set_title(f'Fsv_y (chi-based)  θ={fc.vf_theta:.0f}°')

        for ax in axes:
            ax.set_xlabel('x-index')
            ax.set_ylabel('y-index')

        theta_str = f'{fc.vf_theta:.0f}'
        plt.suptitle(f'Chi diagnostic  iter={iteration}  θ={theta_str}°')
        plt.tight_layout()
        fname = os.path.join(self.IMAGES_SUBDIR,
                             f'chi_diagnostic_theta{theta_str}_iter{iteration:05d}.png')
        plt.savefig(fname, dpi=120)
        plt.close()

    def load_profile(self, runs_dir, label):
        """Reads param_study_runs/<label>/interface_profile.csv (exact
        x -> interface_y data, written by param_study.py from real phi
        data). Returns (xs, ys), or (None, None) if not found."""
        path = os.path.join(runs_dir, label, 'interface_profile.csv')
        if not os.path.exists(path):
            return None, None
        xs, ys = [], []
        with open(path, 'r', newline='') as f:
            for row in csv.DictReader(f):
                xs.append(int(row['x']))
                ys.append(float(row['interface_y']) if row['interface_y'] != '' else float('nan'))
        return xs, ys

    def plot_profile_overlays(self, results_csv, runs_dir, group_param, color_param,
                               out_subdir='overlays', y_lim=(140, 160)):
        """Superimposes each run's interface_profile.csv (from a
        param_study.py sweep), grouped by group_param (one PNG per unique
        value) and colored by color_param (one line per unique value within
        each group). If group_param is constant across the sweep (e.g. a
        single-parameter study), this naturally collapses to one plot."""
        with open(results_csv, 'r', newline='') as f:
            rows = [r for r in csv.DictReader(f) if r['result'] == 'DONE']

        group_vals = sorted({float(r[group_param]) for r in rows})
        out_dir = os.path.join(self.script_dir, out_subdir)
        os.makedirs(out_dir, exist_ok=True)
        cmap = matplotlib.colormaps['viridis']

        saved_paths = []
        for gval in group_vals:
            group = sorted([r for r in rows if float(r[group_param]) == gval],
                            key=lambda r: float(r[color_param]))
            if not group:
                continue

            fig, ax = plt.subplots(figsize=(9, 6))
            for i, r in enumerate(group):
                xs, ys = self.load_profile(runs_dir, r['label'])
                if xs is None:
                    self.debug_log('WARN', f"skipping {r['label']} -- no interface_profile.csv")
                    continue
                color = cmap(i / max(1, len(group) - 1))
                ax.plot(xs, ys, label=f"{color_param}={r[color_param]}",
                        color=color, linewidth=1.8)

            ax.set_xlabel('x-index')
            ax.set_ylabel('interface y-index (phi=0.5 crossing)')
            ax.set_title(f'Superimposed meniscus profiles, {group_param}={gval} (exact)')
            ax.set_ylim(*y_lim)
            ax.legend(title=color_param)
            ax.grid(alpha=0.3)

            out_path = os.path.join(out_dir, f'overlay_{group_param}{gval}.png')
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            self.debug_log('INIT', f'Saved overlay: {out_path}')
            saved_paths.append(out_path)

        return saved_paths
