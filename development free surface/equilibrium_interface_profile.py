# equilibrium_interface_profile.py
import matplotlib.pyplot as plt
import numpy as np

# Parameters
W = 4.0

# z from 0 to W in steps of 0.1
z = np.arange(0, W + 0.01, 0.1)   # +0.01 just to be sure W is included

# Phase-field profile
phi = 0.5 * (1 + np.tanh(2 * z / W))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(z, phi, 'b-', linewidth=2.5)

# Fixed LaTeX (braces around numerator and denominator!)
plt.xlabel(r'$z$', fontsize=14)
plt.ylabel(r'$\phi(z)$', fontsize=14)
plt.title(r'$\phi(z) = \frac{1}{2} \left[1 + \tanh\left(\frac{2z}{W}\right)\right]$  $(W = ' + f'{W}' + ')$',
          fontsize=15, pad=15)

# Alternative (even nicer) using an f-string with raw LaTeX:
# plt.title(fr'$\phi(z) = \frac{{1}}{{2}} \left[1 + \tanh\left(\frac{{2z}}{{W}}\right)\right]$   $(W = {W})$', fontsize=15, pad=15)

plt.grid(True, alpha=0.3)
plt.xlim(0, W)
plt.ylim(-0.05, 1.05)

# Optional annotation
plt.axhline(0.1, color='gray', linestyle='--', alpha=0.7)
plt.axhline(0.9, color='gray', linestyle='--', alpha=0.7)
plt.text(W/2, 0.55, 'Diffuse interface\n(ϕ = 0.1 → 0.9)', 
         ha='center', fontsize=11, 
         bbox=dict(boxstyle="round,pad=0.4", fc="wheat", alpha=0.8))

# tight_layout sometimes fails when complex math is in the title → call it before the title, or skip it
plt.tight_layout()

# If you still get the same error, just comment the line above and uncomment the next one:
# plt.subplots_adjust(top=0.88)

plt.show()