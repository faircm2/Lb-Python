# sympy_inamuro_validation.py
import sympy as sp

# ------------------------------------------------------------------
# Symbols
# ------------------------------------------------------------------
phi, T, a, b, kappa = sp.symbols('phi T a b kappa', real=True, positive=True)
xi, theta, sigma, W = sp.symbols('xi theta sigma W', real=True)

# For 3D Laplacian we need coordinates
x, y, z = sp.symbols('x y z')

# ------------------------------------------------------------------
# 1. Inamuro's bulk free-energy density ψ (Eq. 11)
# ------------------------------------------------------------------
psi_bulk = phi * T * sp.log(phi / (1 - b*phi)) - a * phi**2

# ------------------------------------------------------------------
# 2. Chemical potential (variational derivative)
# ------------------------------------------------------------------
# ∂ψ/∂ϕ
dpsi_dphi = sp.diff(psi_bulk, phi)

# Laplacian of ϕ (manually defined)
laplacian_phi = sp.diff(phi, x, 2) + sp.diff(phi, y, 2) + sp.diff(phi, z, 2)

mu_c = dpsi_dphi - kappa * laplacian_phi

print("=== Inamuro chemical potential μ_c ===")
sp.pprint(mu_c)
print("\nSimplified version:")
sp.pprint(mu_c.subs(laplacian_phi, sp.Symbol('∇²ϕ')))

print("\n→ Explicit dψ/dϕ term:")
sp.pprint(dpsi_dphi.simplify())