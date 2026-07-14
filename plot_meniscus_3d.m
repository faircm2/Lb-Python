% Plot meniscus rise/fall vs tau_g and tau_f as a 3D surface.
% Reads param_study_phase1_tau_results.csv (must be in the same folder).

data = readtable('param_study_phase1_tau_results.csv');

tau_g_vals = unique(data.tau_g);
tau_f_vals = unique(data.tau_f);

grid = NaN(numel(tau_g_vals), numel(tau_f_vals));
for k = 1:height(data)
    if strcmp(data.result{k}, 'DONE')
        i = find(tau_g_vals == data.tau_g(k));
        j = find(tau_f_vals == data.tau_f(k));
        grid(i, j) = data.meniscus_rise(k);
    end
end

[TF, TG] = meshgrid(tau_f_vals, tau_g_vals);  % note: rows=tau_g, cols=tau_f

figure('Position', [100, 100, 900, 700]);
surf(TG, TF, grid, 'EdgeColor', 'k', 'LineWidth', 0.3);
colormap(parula);   % MATLAB's built-in default; swap for colormap(viridis) if you
                     % have that function on your path (not built into plain MATLAB)
colorbar;
shading interp;

xlabel('tau_g');
ylabel('tau_f');
zlabel('meniscus rise/fall (lattice units)');
title('Meniscus rise/fall vs tau_g and tau_f (vf\_theta=60)');

view(-40, 25);  % roughly matches the Python plot's viewing angle
grid on;
