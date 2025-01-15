import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import ufl
import dolfinx
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import json

import fenicsx_growth.geometry as geo
import fenicsx_growth.ddf as ddf
import fenicsx_growth.postprocessing as pp
import fenicsx_growth.hyperelastic_models as psi
import fenicsx_growth.growth_laws as gl

# region
"""Create Geometry"""
x_min, x_max, Nx = 0, 1, 2
y_min, y_max, Ny = 0, 1, 2
z_min, z_max, Nz = 0, 1, 2
xs = np.linspace(x_min, x_max, Nx)
ys = np.linspace(y_min, y_max, Ny)
zs = np.linspace(z_min, z_max, Nz)
X = [xs, ys, zs]
mesh = geo.create_box(X)

"""Get Functions"""
vector_space, u, v = ddf.get_vector_functions(mesh)

"""Boundary Conditions"""
"""Dirichlet"""
x_left_x = (lambda x: np.isclose(x[0], x_min), 0, dolfinx.default_scalar_type(0))
x_right_x = (lambda x: np.isclose(x[0], x_max), 0, dolfinx.default_scalar_type(0.1))
y_left_y = (lambda x: np.isclose(x[1], y_min), 1, dolfinx.default_scalar_type(0))
y_right_y = (lambda x: np.isclose(x[1], y_max), 1, dolfinx.default_scalar_type(0))
z_left_z = (lambda x: np.isclose(x[2], z_min), 2, dolfinx.default_scalar_type(0))
z_right = (lambda x: np.isclose(x[2], z_max), 2, dolfinx.default_scalar_type(0))

bc_values = [x_left_x, y_left_y, z_left_z, x_right_x]

"""Kinematics"""
I = ufl.Identity(len(u))
F = ufl.variable(I + ufl.grad(u))

F_g_prev_function, F_g_prev_expression, E_e_function, E_e_expression, F_e = gl.KFR(
    mesh, F
)
J_e = ufl.variable(ufl.det(F_e))
C_e = F_e.T * F_e

"""Constants"""
mu = dolfinx.default_scalar_type(15)
kappa = dolfinx.default_scalar_type(1e3)

"""Create compressible strain energy function"""
psi_inc = psi.neohookean(mu, C_e)
psi_comp = psi.comp2(kappa, J_e)
psi_ = psi_inc + psi_comp
P = ufl.diff(psi_, F_e)

"""Create weak form"""
weak_form = ufl.inner(ufl.grad(v), P) * ufl.dx(metadata={"quadrature_degree": 8})

"""Apply Boundary Conditions"""
bcs, _ = ddf.dirichlet_injection(mesh, bc_values, vector_space)

"""Assemble FEniCS solver"""
problem = NonlinearProblem(weak_form, u, bcs)
solver = NewtonSolver(mesh.comm, problem)

us = []
"""Solve once to get set point values etc."""

problem = NonlinearProblem(weak_form, u, bcs)
solver = NewtonSolver(mesh.comm, problem)

us = []
F_g_f_prev = []
F_g_c_prev = []
F_e_f = []
F_e_c = []
"""Solve The Problem"""
N = 20  # Number of time steps
time = np.linspace(0, 20, N + 1)  # number of days measured in N steps

for i in range(0, N + 1):
    if i % (N / 100) == 0:
        print(f"Percent finished: {round(i/N*100)} %")
        # breakpoint()

    solver.solve(u)  # Solve the problem

    F_g_prev_function.interpolate(F_g_prev_expression)  # Update total growth tensor
    E_e_function.interpolate(E_e_expression)  # Update elastic deformation tensor

    F_g_f_prev.append(ddf.eval_expression(F_g_prev_function[0, 0], mesh)[0, 0])
    F_g_c_prev.append(ddf.eval_expression(F_g_prev_function[1, 1], mesh)[0, 0])
    F_e_f.append(ddf.eval_expression(F_e[0, 0], mesh)[0, 0])
    F_e_c.append(ddf.eval_expression(F_e[1, 1], mesh)[0, 0])


data_to_save = {
    "data_lists": [[F_g_f_prev], [F_g_c_prev], [F_e_f], [F_e_c]],
    "titles": [
        "Growth in fiber direction",
        "Growth in crossfiber direction",
        "Stretch in fiber direction",
        "Stretch in crossfiber direction",
    ],
    "ylabels": ["Distance", "Distance", "Stretch ratio", "Stretch ratio"],
}
with open("../Plots/10p_stretch_crossfiber_KFR.json", "w") as f:
    json.dump(data_to_save, f)

# Load data from JSON file
with open("../Plots/10p_stretch_crossfiber_KFR.json", "r") as f:
    loaded_data = json.load(f)

data_lists = loaded_data["data_lists"]
titles = loaded_data["titles"]
ylabels = loaded_data["ylabels"]

# Create figure with GridSpec
fig = plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(2, 2)
axes = []

for i in range(4):
    ax = pp.make_plot(fig, gs, i, data_lists[i], titles[i], ylabels[i], time)

    axes.append(ax)

# Adjust layout
plt.tight_layout()

# Create a single legend for all subplots
lines = axes[0].get_lines()

plt.subplots_adjust(bottom=0.1)

# Save the figure
plt.savefig("../Plots/10p_stretch_crossfiber_KFR.png", dpi=300, bbox_inches="tight")
plt.close()

print("Plot saved as 10p_stretch_crossfiber_KFR.png")
