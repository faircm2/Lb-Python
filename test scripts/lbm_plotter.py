import os

import matplotlib.pyplot as plt
import numpy as np


class LBMPlotter:
    def __init__(self, script_dir, script_filename, use_case_tag, images_subdir,
                 total_iterations, filename_padding_width, debug_log):
        self.script_dir = script_dir
        self.SCRIPT_FILENAME = script_filename
        self.USE_CASE_TAG = use_case_tag
        self.IMAGES_SUBDIR = images_subdir
        self.TOTAL_ITERATIONS = total_iterations
        self.FILENAME_PADDING_WIDTH = filename_padding_width
        self.debug_log = debug_log

    # -------------------------------------------------------------
    # Amplitude Plot
    # -------------------------------------------------------------
    def amplitude_plot0(self, ax1, u_full_range, listIterations, axis, xlabel, ylabel, title,
                       nx, ny, angle=45, font_size=10):
        fig_standalone = plt.figure(figsize=(10, 6))
        ax_standalone = fig_standalone.add_subplot(111)
        for iteration, combined_u_ckl in u_full_range.items():
            u = combined_u_ckl[nx, 1:ny + 1]
            ax_standalone.plot(u, axis, label=f"t={iteration}")

        self._style_lineplot(ax_standalone, xlabel, ylabel, title, angle, font_size)
        self._save_plot(fig_standalone, f"{title}_{max(listIterations[-1], self.total_iterations)}.png")

        # subplot version
        for iteration, combined_u_ckl in u_full_range.items():
            u = combined_u_ckl[nx, 1:ny + 1]
            ax1.plot(u, axis, label=f"t={iteration}")
        self._style_lineplot(ax1, xlabel, ylabel, title, angle, font_size)

    #amplitude_plot
    def amplitude_plot(self, ax1, u_full_range, listIterations, axis, xlabel, ylabel, title, nx, ny, angle=45, font_size=10):
        # Create standalone figure
        fig_standalone = plt.figure(figsize=(10, 6))
        ax_standalone = fig_standalone.add_subplot(111)
        for iteration, combined_u_ckl in u_full_range.items():
            u = combined_u_ckl[nx, 1:ny + 1]
            ax_standalone.plot(u, axis, label=f"t={iteration}")
        ax_standalone.grid()
        ax_standalone.set_xlabel(xlabel)
        ax_standalone.set_ylabel(ylabel)
        ax_standalone.set_title(title)
        ax_standalone.set_ylim(-1, 51)
        ax_standalone.set_yticks(np.arange(0, 51, 10))
        ax_standalone.legend(ncol=1, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=font_size)
        plt.setp(ax_standalone.get_xticklabels(), rotation=angle, ha='right', fontsize=font_size)
        ax_standalone.tick_params(axis='x', labelsize=font_size)
        ax_standalone.margins(x=0, y=0)
        x_min, x_max = ax_standalone.get_xlim()
        ax_standalone.set_xlim(x_min, x_max + 0.1*(x_max-x_min))
        
        # Save standalone plot with unique filename
        images_dir = os.path.join(self.script_dir, "FreesurfaceImages")
        os.makedirs(images_dir, exist_ok=True)
        # Differentiate u_x and u_y explicitly
        filename = f"{self.SCRIPT_FILENAME}_{self.USE_CASE_TAG}_{title}_{max(listIterations[-1], self.TOTAL_ITERATIONS):0{self.FILENAME_PADDING_WIDTH}d}.png"
        save_path = os.path.join(images_dir, filename)
        fig_standalone.savefig(save_path, dpi=300, bbox_inches='tight')
        self.debug_log('INIT', 'Saved amplitude plot: %s', save_path)
        plt.close(fig_standalone)

        # Plot on provided ax for subplot grid
        for iteration, combined_u_ckl in u_full_range.items():
            u = combined_u_ckl[nx, 1:ny + 1]
            ax1.plot(u, axis, label=f"t={iteration}")
        ax1.grid()
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(title)
        ax1.set_ylim(-1, 51)
        ax1.set_yticks(np.arange(0, 51, 10))
        ax1.legend(ncol=1, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=font_size)
        plt.setp(ax1.get_xticklabels(), rotation=angle, ha='right', fontsize=font_size)
        ax1.tick_params(axis='x', labelsize=font_size)
        ax1.margins(x=0, y=0)
        x_min, x_max = ax1.get_xlim()
        ax1.set_xlim(x_min, x_max + 0.1*(x_max-x_min))


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
        filename = f"{self.SCRIPT_FILENAME}_{self.USE_CASE_TAG}_density_profile_{iteration:0{self.FILENAME_PADDING_WIDTH}d}.png"
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
        filename = f"{self.SCRIPT_FILENAME}_{self.USE_CASE_TAG}_density_profile_transition_{iteration:0{self.FILENAME_PADDING_WIDTH}d}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)

        ax1.figure.savefig(save_path, dpi=300, bbox_inches='tight')
        self.debug_log('INIT', 'Saved density profile transition: %s', save_path)


    # Plot density profiles saved at given iterations
    def density_profiles(self, ax, density_slices, x__position, nx, ny):
        """Compact overlay of density slices with right legend."""
        if not density_slices: return
        y_coords = np.arange(0, ny + 2)
        iterations, slices = zip(*density_slices)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for idx, (it, rho_slice) in enumerate(zip(iterations, slices)):
            ax.plot(rho_slice, y_coords, color=colors[idx % 10], label=f"Iter {it}")
        ax.set(xlabel="Density [rho]", ylabel="y-axis", title=f"Evolution at x={x__position}",
            xlim=(-1, 52), xticks=np.linspace(-1, 52, 5), ylim=(0, ny + 1), yticks=np.linspace(0, ny + 1, 6))
        ax.grid()
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize='small')
        ax.figure.tight_layout()

        # Save in same directory as the script
        filename = f"{self.SCRIPT_FILENAME}_{self.USE_CASE_TAG}_density_profiles_{self.iteration:0{self.FILENAME_PADDING_WIDTH}d}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)

        ax.figure.savefig(save_path, dpi=300, bbox_inches='tight')
        self.debug_log('INIT', 'Saved density profiles: %s', save_path)


    #2D Density map
    def density_mapExt(self, ax, full_range, min, max, title, _iteration):
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
        filename = f"{self.SCRIPT_FILENAME}_{self.USE_CASE_TAG}_density_mapExt_{self.iteration:0{self.FILENAME_PADDING_WIDTH}d}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)

        ax.figure.savefig(save_path, dpi=300, bbox_inches='tight')
        self.debug_log('INIT', 'Saved density_mapExt: %s', save_path)


    def density_map_standalone(self, full_range, min_val, max_val, title, _iteration, z_index=None):
        """
        Save a 2D density map plot from a 3D density field to a PNG file with zero-padded iteration number.

        Args:
            full_range: 3D numpy array of density values (nx, ny, nz), e.g., (202, 52, 52).
            min_val: Lower bound for colorbar (float).
            max_val: Upper bound for colorbar (float).
            title: Plot title (str).
            _iteration: Current iteration number (int).
            z_index: Index for z-slice (int, default: nz//2 for midpoint).

        Returns:
            None (saves PNG).
        """
        import os

        import matplotlib.pyplot as plt

        # Handle 3D input by extracting 2D x-y slice
        if len(full_range.shape) != 3:
            raise ValueError(f"Expected 3D array with shape (nx, ny, nz), got {full_range.shape}")
        
        nx, ny, nz = full_range.shape
        if z_index is None:
            z_index = nz // 2  # Default to midpoint
        if not (0 <= z_index < nz):
            raise ValueError(f"z_index {z_index} out of bounds for nz={nz}")

        slice_2d = full_range[:, :, z_index]  # Shape (nx, ny), e.g., (202, 52)

        self.debug_log('FIELD', 'Debug density_map: min=%.6f, max=%.6f, rho_out=%.6f, rho_in=%.6f', 
                np.min(slice_2d), np.max(slice_2d), min_val, max_val)

        # Create a new figure and axes
        fig, ax = plt.subplots(figsize=(6, 6))  # Original size

        # Plot the image
        im = ax.imshow(slice_2d.T, interpolation='nearest', origin='lower',
                    cmap='viridis', vmin=min_val, vmax=max_val)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title(title)
        ax.margins(x=0, y=0)
        ax.set_aspect('auto')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(min_val, max_val, 3))  # min, mid, max
        cbar.set_label("Density")

        # Save PNG with dynamic zero-padding
        filename = f"{self.SCRIPT_FILENAME}_{self.USE_CASE_TAG}_density_map_standalone_{_iteration:0{self.FILENAME_PADDING_WIDTH}d}_Z{z_index}.png"

        save_path = os.path.join(self.IMAGES_SUBDIR, filename)

        ax.figure.savefig(save_path, dpi=300, bbox_inches='tight')
        self.debug_log('INIT', 'Saved density_map_standalone: %s', save_path)
        plt.close(fig)  # Close to prevent memory leaks
    

    # 2D Velocity map
    def velocity_map(self, ax, u_magnitude, _iteration, title):
        # Create standalone figure
        fig_standalone = plt.figure(figsize=(8, 5))
        ax_standalone = fig_standalone.add_subplot(111)
        im = ax_standalone.imshow(u_magnitude.T, cmap='viridis', origin='lower')
        ax_standalone.set_xlabel('x')
        ax_standalone.set_ylabel('y')
        ax_standalone.set_title(title)
        plt.colorbar(im, ax=ax_standalone, label='Velocity')
        
        # Save standalone plot
        images_dir = os.path.join(self.script_dir, "FreesurfaceImages")
        os.makedirs(images_dir, exist_ok=True)
        # Use TOTAL_ITERATIONS for consistency if _iteration is negative
        iteration_str = f"{max(_iteration, self.TOTAL_ITERATIONS):0{self.FILENAME_PADDING_WIDTH}d}"
        # Simplify title for filename (e.g., 'Velocity_ux' or 'Velocity_uy')
        simplified_title = title.replace('Velocity [u$_x$] map', 'velocity_ux').replace('Velocity [u$_y$] map', 'velocity_uy')
        filename = f"{self.SCRIPT_FILENAME}_{self.USE_CASE_TAG}_{simplified_title}_{iteration_str}.png"
        save_path = os.path.join(images_dir, filename)
        fig_standalone.savefig(save_path, dpi=300, bbox_inches='tight')
        self.debug_log('INIT', 'Saved velocity map: %s', save_path)
        plt.close(fig_standalone)

        # Plot on provided ax for subplot grid
        im = ax.imshow(u_magnitude.T, cmap='viridis', origin='lower')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Velocity')


    def filter_u_ckl_fullrange(self, velocities_dict, iterationsOfInterest):
        filtered_velocities = {iteration: velocities_dict[iteration] for iteration in iterationsOfInterest if iteration in velocities_dict}

        return filtered_velocities   


    def plot_bounds_ext(self, results, context, ax=None, series_labels=None, k=None, script_filename=None):
        if results is None or not isinstance(results, (list, tuple)) or not results or len(results[0]) < 2:
            raise ValueError("results must be non-empty list of (iteration, values)")
        if isinstance(results, np.ndarray): results = results.tolist()
        
        iterations = [r[0] for r in results]
        data_series = list(zip(*[r[1:] for r in results]))
        n_series = len(data_series)
        series_labels = series_labels or [f"{context}_{i+1}" for i in range(n_series)]
        
        ylabel, title = context, f"{context} vs Iteration"
        
        # Create standalone figure
        fig_standalone = plt.figure(figsize=(8, 5))
        for series, label in zip(data_series, series_labels):
            plt.plot(iterations, series, label=label)
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        # Save standalone plot with unique context
        if script_filename is None: script_filename = self.SCRIPT_FILENAME
        filename = f"{script_filename}_{self.USE_CASE_TAG}_{context.replace(' ', '_')}_{self.TOTAL_ITERATIONS:0{self.FILENAME_PADDING_WIDTH}d}.png"
        images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FreesurfaceImages")
        os.makedirs(images_dir, exist_ok=True)
        save_path = os.path.join(images_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig_standalone)
        self.debug_log('INIT', 'Saved plot_bounds_ext: %s', save_path)
        
        # Plot on provided ax
        if ax is not None:
            plt.sca(ax)
            for series, label in zip(data_series, series_labels):
                plt.plot(iterations, series, label=label)
            plt.xlabel("Iteration")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
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
        filename = f"{self.SCRIPT_FILENAME}_{self.USE_CASE_TAG}_{_filename}_{self.TOTAL_ITERATIONS:0{self.FILENAME_PADDING_WIDTH}d}.png"
        images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FreesurfaceImages")
        os.makedirs(images_dir, exist_ok=True)
        save_path = os.path.join(images_dir, filename)
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


    def save_phi_snapshot(self, _phi, iteration, phi_star_G, phi_star_L, z_index=None):
        """
        Save a snapshot plot of a 2D slice of the 3D order parameter phi and print min/max/mean stats.

        Args:
            _phi: 3D numpy array of phi values (nx, ny, nz), e.g., (202, 52, 52).
            iteration: Current iteration number (int).
            phi_star_G: Lower bound for colorbar (float).
            phi_star_L: Upper bound for colorbar (float).
            z_index: Index for z-slice (int, default: nz//2 for midpoint).

        Returns:
            None (saves PNG and prints stats).
        """
        # Handle 3D input by extracting 2D x-y slice
        if len(_phi.shape) != 3:
            raise ValueError(f"Expected 3D array with shape (nx, ny, nz), got {_phi.shape}")
        
        nx, ny, nz = _phi.shape
        if z_index is None:
            z_index = nz // 2  # Default to midpoint
        if not (0 <= z_index < nz):
            raise ValueError(f"z_index {z_index} out of bounds for nz={nz}")

        slice_2d = _phi[:, :, z_index]  # Shape (nx, ny), e.g., (202, 52)

        # Create plot
        plt.figure(figsize=(8, 6))
        im = plt.imshow(slice_2d.T, origin='lower', cmap='RdBu', vmin=phi_star_G, vmax=phi_star_L)
        plt.colorbar(im, label='phi')
        plt.title(f'Order parameter phi at iteration {iteration}')
        plt.xlabel('x-index')
        plt.ylabel('y-index')

        # Save PNG with USE_CASE_TAG
        filename = f"{self.SCRIPT_FILENAME}_{self.USE_CASE_TAG}_phi_snapshot_iter_{iteration:0{self.FILENAME_PADDING_WIDTH}d}_Z{z_index}.png"
        save_path = os.path.join(self.IMAGES_SUBDIR, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Print stats for the 2D slice
        phi_min = np.min(slice_2d)
        phi_max = np.max(slice_2d)
        phi_mean = np.mean(slice_2d)
        self.debug_log('ITER', 'phi (slice z=%d) at iter %d: min=%.3f, max=%.3f, mean=%.3f', 
                z_index, iteration, phi_min, phi_max, phi_mean)
        # Optionally log 3D stats
        self.debug_log('ITER', 'phi (3D) at iter %d: min=%.3f, max=%.3f, mean=%.3f', 
                iteration, np.min(_phi), np.max(_phi), np.mean(_phi))
        self.debug_log('INIT', 'Saved phi snapshot: %s', save_path)
