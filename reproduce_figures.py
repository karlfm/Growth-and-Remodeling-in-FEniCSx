import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import ufl
import dolfinx
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import basix
import json

import fenicsx_growth.geometry as geo
import fenicsx_growth.ddf as ddf
import fenicsx_growth.postprocessing as pp
import fenicsx_growth.hyperelastic_models as psi
import fenicsx_growth.growth_laws as gl


def get_parser():
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="Plots",
        help="Output folder for plots",
    )
    parser.add_argument(
        "-n",
        "-num-steps",
        type=int,
        dest="N",
        default=2000,
        help="Number of time steps",
    )
    parser.add_argument(
        "-d",
        "--num-days",
        dest="number_of_days",
        type=int,
        default=20,
        help="Number of days to simulate",
    )
    return parser


def main(output: Path = Path("Plots"), N: int = 2000, number_of_days: int = 20):
    """Create Geometry"""
    x_min, x_max, Nx = 0, 1, 2
    y_min, y_max, Ny = 0, 1, 2
    z_min, z_max, Nz = 0, 1, 2
    xs = np.linspace(x_min, x_max, Nx)
    ys = np.linspace(y_min, y_max, Ny)
    zs = np.linspace(z_min, z_max, Nz)
    X = [xs, ys, zs]
    mesh_LT2 = geo.create_box(X)
    mesh_KFR = geo.create_box(X)
    mesh_GEG = geo.create_box(X)
    mesh_GCG = geo.create_box(X)
    mesh_KOM = geo.create_box(X)

    """Get Functions"""
    vector_space_LT2, u_LT2, v_LT2 = ddf.get_vector_functions(mesh_LT2)
    vector_space_KFR, u_KFR, v_KFR = ddf.get_vector_functions(mesh_KFR)
    vector_space_GEG, u_GEG, v_GEG = ddf.get_vector_functions(mesh_GEG)
    vector_space_GCG, u_GCG, v_GCG = ddf.get_vector_functions(mesh_GCG)
    vector_space_KOM, u_KOM, v_KOM = ddf.get_vector_functions(mesh_KOM)

    """Boundary Conditions"""
    # Dirichlet
    x_left_x = (lambda x: np.isclose(x[0], x_min), 0, dolfinx.default_scalar_type(0))
    x_right_x = (
        lambda x: np.isclose(x[0], x_max),
        0,
        dolfinx.default_scalar_type(0.1),
    )  # change this to -0.1 if you want compression
    y_left_y = (lambda x: np.isclose(x[1], y_min), 1, dolfinx.default_scalar_type(0))
    # y_right_y = (lambda x: np.isclose(x[1], y_max), 1, dolfinx.default_scalar_type(0))
    z_left_z = (lambda x: np.isclose(x[2], z_min), 2, dolfinx.default_scalar_type(0))
    # z_right = (lambda x: np.isclose(x[2], z_max), 2, dolfinx.default_scalar_type(0))

    bc_values = [x_left_x, y_left_y, z_left_z, x_right_x]

    """Kinematics"""
    F_LT2 = ufl.variable(ufl.Identity(len(u_LT2)) + ufl.grad(u_LT2))
    F_KFR = ufl.variable(ufl.Identity(len(u_KFR)) + ufl.grad(u_KFR))
    F_GEG = ufl.variable(ufl.Identity(len(u_GEG)) + ufl.grad(u_GEG))
    F_GCG = ufl.variable(ufl.Identity(len(u_GCG)) + ufl.grad(u_GCG))
    F_KOM = ufl.variable(ufl.Identity(len(u_KOM)) + ufl.grad(u_KOM))

    ###########################################################################################################################################################################
    F_g_tot_function_LT2, F_g_tot_expression_LT2, sigma_function_LT2, F_e_LT2 = gl.LT2(
        mesh_LT2, F_LT2
    )
    (
        F_g_tot_function_KFR,
        F_g_tot_expression_KFR,
        E_e_function_KFR,
        E_e_expression_KFR,
        F_e_KFR,
    ) = gl.KFR(mesh_KFR, F_KFR)
    (
        F_g_tot_function_GEG,
        F_g_tot_expression_GEG,
        E_e_function_GEG,
        E_e_expression_GEG,
        F_e_GEG,
    ) = gl.GEG(mesh_GEG, F_GEG)
    F_g_prev_function_GCG, F_g_prev_expression_GCG, mandel_function_GCG, F_e_GCG = (
        gl.GCG(mesh_GCG, F_GCG)
    )
    (
        F_g_tot_function_KOM,
        F_g_tot_expression_KOM,
        E_e_function_KOM,
        E_e_expression_KOM,
        F_e_KOM,
    ) = gl.KOM(mesh_KOM, F_KOM)

    J_e_LT2 = ufl.det(F_e_LT2)
    F_e_bar_LT2 = F_e_LT2 * pow(J_e_LT2, -1 / 3)
    C_e_bar_LT2 = F_e_bar_LT2.T * F_e_bar_LT2
    J_e_KFR = ufl.det(F_e_KFR)
    F_e_bar_KFR = F_e_KFR * pow(J_e_KFR, -1 / 3)
    C_e_bar_KFR = F_e_bar_KFR.T * F_e_bar_KFR
    J_e_GEG = ufl.det(F_e_GEG)
    F_e_bar_GEG = F_e_GEG * pow(J_e_GEG, -1 / 3)
    C_e_bar_GEG = F_e_bar_GEG.T * F_e_bar_GEG
    J_e_GCG = ufl.det(F_e_GCG)
    F_e_bar_GCG = F_e_GCG * pow(J_e_GCG, -1 / 3)
    C_e_bar_GCG = F_e_bar_GCG.T * F_e_bar_GCG
    J_e_KOM = ufl.det(F_e_KOM)
    F_e_bar_KOM = F_e_KOM * pow(J_e_KOM, -1 / 3)
    C_e_bar_KOM = F_e_bar_KOM.T * F_e_bar_KOM

    """Constants"""
    mu = dolfinx.default_scalar_type(15)
    kappa = dolfinx.default_scalar_type(1e3)

    """Create compressible strain energy function"""
    psi_LT2 = psi.neohookean(mu, C_e_bar_LT2) + psi.comp2(kappa, J_e_LT2)
    P_LT2 = ufl.diff(psi_LT2, F_e_LT2)
    tensor_space_LT2 = dolfinx.fem.functionspace(
        mesh_LT2,
        basix.ufl.element(
            family="DG", cell=str(mesh_LT2.ufl_cell()), degree=1, shape=(3, 3)
        ),
    )
    sigma_expression_LT2 = dolfinx.fem.Expression(
        P_LT2 * F_e_LT2.T / J_e_LT2, tensor_space_LT2.element.interpolation_points()
    )

    psi_KFR = psi.neohookean(mu, C_e_bar_KFR) + psi.comp2(kappa, J_e_KFR)
    P_KFR = ufl.diff(psi_KFR, F_e_KFR)
    psi_GEG = psi.neohookean(mu, C_e_bar_GEG) + psi.comp2(kappa, J_e_GEG)
    P_GEG = ufl.diff(psi_GEG, F_e_GEG)
    psi_GCG = psi.neohookean(mu, C_e_bar_GCG) + psi.comp2(kappa, J_e_GCG)
    P_GCG = ufl.diff(psi_GCG, F_e_GCG)
    tensor_space_GCG = dolfinx.fem.functionspace(
        mesh_GCG,
        basix.ufl.element(
            family="DG", cell=str(mesh_GCG.ufl_cell()), degree=1, shape=(3, 3)
        ),
    )
    mandel_expression_GCG = dolfinx.fem.Expression(
        F_e_GCG.T * P_GCG, tensor_space_GCG.element.interpolation_points()
    )

    psi_KOM = psi.neohookean(mu, C_e_bar_KOM) + psi.comp2(kappa, J_e_KOM)
    P_KOM = ufl.diff(psi_KOM, F_e_KOM)

    """Create weak form"""
    weak_form_LT2 = ufl.inner(ufl.grad(v_LT2), P_LT2) * ufl.dx(
        metadata={"quadrature_degree": 8}
    )
    weak_form_KFR = ufl.inner(ufl.grad(v_KFR), P_KFR) * ufl.dx(
        metadata={"quadrature_degree": 8}
    )
    weak_form_GEG = ufl.inner(ufl.grad(v_GEG), P_GEG) * ufl.dx(
        metadata={"quadrature_degree": 8}
    )
    weak_form_GCG = ufl.inner(ufl.grad(v_GCG), P_GCG) * ufl.dx(
        metadata={"quadrature_degree": 8}
    )
    weak_form_KOM = ufl.inner(ufl.grad(v_KOM), P_KOM) * ufl.dx(
        metadata={"quadrature_degree": 8}
    )

    """Apply Boundary Conditions"""
    bcs_LT2, _ = ddf.dirichlet_injection(mesh_LT2, bc_values, vector_space_LT2)
    bcs_KFR, _ = ddf.dirichlet_injection(mesh_KFR, bc_values, vector_space_KFR)
    bcs_GEG, _ = ddf.dirichlet_injection(mesh_GEG, bc_values, vector_space_GEG)
    bcs_GCG, _ = ddf.dirichlet_injection(mesh_GCG, bc_values, vector_space_GCG)
    bcs_KOM, _ = ddf.dirichlet_injection(mesh_KOM, bc_values, vector_space_KOM)

    """Assemble FEniCS solver"""
    problem_LT2 = NonlinearProblem(weak_form_LT2, u_LT2, bcs_LT2)
    problem_KFR = NonlinearProblem(weak_form_KFR, u_KFR, bcs_KFR)
    problem_GEG = NonlinearProblem(weak_form_GEG, u_GEG, bcs_GEG)
    problem_GCG = NonlinearProblem(weak_form_GCG, u_GCG, bcs_GCG)
    problem_KOM = NonlinearProblem(weak_form_KOM, u_KOM, bcs_KOM)

    solver_LT2 = NewtonSolver(mesh_LT2.comm, problem_LT2)
    solver_KFR = NewtonSolver(mesh_KFR.comm, problem_KFR)
    solver_GEG = NewtonSolver(mesh_GEG.comm, problem_GEG)
    solver_GCG = NewtonSolver(mesh_GCG.comm, problem_GCG)
    solver_KOM = NewtonSolver(mesh_KOM.comm, problem_KOM)

    # Preallocate lists for postprocessing
    us_LT2 = []
    F_g_f_tot_LT2 = []
    F_g_c_tot_LT2 = []
    F_e_f_LT2 = []
    F_e_c_LT2 = []
    us_KFR = []
    F_g_f_tot_KFR = []
    F_g_c_tot_KFR = []
    F_e_f_KFR = []
    F_e_c_KFR = []
    us_GEG = []
    F_g_f_tot_GEG = []
    F_g_c_tot_GEG = []
    F_e_f_GEG = []
    F_e_c_GEG = []
    us_GCG = []
    F_g_f_tot_GCG = []
    F_g_c_tot_GCG = []
    F_e_f_GCG = []
    F_e_c_GCG = []
    us_KOM = []
    F_g_f_tot_KOM = []
    F_g_c_tot_KOM = []
    F_e_f_KOM = []
    F_e_c_KOM = []

    """Solve The Problem"""
    time = np.linspace(0, number_of_days, N + 1)  # number of days measured in N steps
    for i in range(0, N + 1):
        if i % (N / 100) == 0:  # (N/100)
            print(f"Time step {round(i/N*100)} %")
            # print(f"Time step {i}/{N}")
            # breakpoint()

        solver_LT2.solve(u_LT2)
        solver_KFR.solve(u_KFR)
        solver_GEG.solve(u_GEG)
        solver_GCG.solve(u_GCG)
        solver_KOM.solve(u_KOM)

        F_g_tot_function_LT2.interpolate(F_g_tot_expression_LT2)
        sigma_function_LT2.interpolate(sigma_expression_LT2)
        F_g_tot_function_KFR.interpolate(F_g_tot_expression_KFR)
        E_e_function_KFR.interpolate(E_e_expression_KFR)
        F_g_tot_function_GEG.interpolate(F_g_tot_expression_GEG)
        E_e_function_GEG.interpolate(E_e_expression_GEG)
        F_g_prev_function_GCG.interpolate(F_g_prev_expression_GCG)
        mandel_function_GCG.interpolate(mandel_expression_GCG)
        F_g_tot_function_KOM.interpolate(F_g_tot_expression_KOM)
        E_e_function_KOM.interpolate(E_e_expression_KOM)

        # Tabulate values for postprocessing
        u_LT2_new = u_LT2.copy()
        us_LT2.append(u_LT2_new)
        u_KFR_new = u_KFR.copy()
        us_KFR.append(u_KFR_new)
        u_GEG_new = u_GEG.copy()
        us_GEG.append(u_GEG_new)
        u_GCG_new = u_GCG.copy()
        us_GCG.append(u_GCG_new)
        u_KOM_new = u_KOM.copy()
        us_KOM.append(u_KOM_new)

        F_g_f_tot_LT2.append(
            ddf.eval_expression(F_g_tot_function_LT2[0, 0], mesh_LT2)[0, 0]
        )
        F_g_f_tot_KFR.append(
            ddf.eval_expression(F_g_tot_function_KFR[0, 0], mesh_KFR)[0, 0]
        )
        F_g_f_tot_GEG.append(
            ddf.eval_expression(F_g_tot_function_GEG[0, 0], mesh_GEG)[0, 0]
        )
        F_g_f_tot_GCG.append(
            ddf.eval_expression(F_g_prev_function_GCG[0, 0], mesh_GCG)[0, 0]
        )
        F_g_f_tot_KOM.append(
            ddf.eval_expression(F_g_tot_function_KOM[0, 0], mesh_KOM)[0, 0]
        )

        F_g_c_tot_LT2.append(
            ddf.eval_expression(F_g_tot_function_LT2[1, 1], mesh_LT2)[0, 0]
        )
        F_g_c_tot_KFR.append(
            ddf.eval_expression(F_g_tot_function_KFR[1, 1], mesh_KFR)[0, 0]
        )
        F_g_c_tot_GEG.append(
            ddf.eval_expression(F_g_tot_function_GEG[1, 1], mesh_GEG)[0, 0]
        )
        F_g_c_tot_GCG.append(
            ddf.eval_expression(F_g_prev_function_GCG[1, 1], mesh_GCG)[0, 0]
        )
        F_g_c_tot_KOM.append(
            ddf.eval_expression(F_g_tot_function_KOM[1, 1], mesh_KOM)[0, 0]
        )

        F_e_f_LT2.append(ddf.eval_expression(F_e_LT2[0, 0], mesh_LT2)[0, 0])
        F_e_f_KFR.append(ddf.eval_expression(F_e_KFR[0, 0], mesh_KFR)[0, 0])
        F_e_f_GEG.append(ddf.eval_expression(F_e_GEG[0, 0], mesh_GEG)[0, 0])
        F_e_f_GCG.append(ddf.eval_expression(F_e_GCG[0, 0], mesh_GCG)[0, 0])
        F_e_f_KOM.append(ddf.eval_expression(F_e_KOM[0, 0], mesh_KOM)[0, 0])

        F_e_c_LT2.append(ddf.eval_expression(F_e_LT2[1, 1], mesh_LT2)[0, 0])
        F_e_c_KFR.append(ddf.eval_expression(F_e_KFR[1, 1], mesh_KFR)[0, 0])
        F_e_c_GEG.append(ddf.eval_expression(F_e_GEG[1, 1], mesh_GEG)[0, 0])
        F_e_c_GCG.append(ddf.eval_expression(F_e_GCG[1, 1], mesh_GCG)[0, 0])
        F_e_c_KOM.append(ddf.eval_expression(F_e_KOM[1, 1], mesh_KOM)[0, 0])

    """Write to file to plot in Desmos and Paraview"""
    lists_to_write = {
        "F_{gffLT2}": F_g_f_tot_LT2,
        "F_{gccLT2}": F_g_c_tot_LT2,
        "F_{gffKFR}": F_g_f_tot_KFR,
        "F_{gccKFR}": F_g_c_tot_KFR,
        "F_{gffGEG}": F_g_f_tot_GEG,
        "F_{gccGEG}": F_g_c_tot_GEG,
        "F_{gffGCG}": F_g_f_tot_GCG,
        "F_{gccGCG}": F_g_c_tot_GCG,
        "F_{gffKOM}": F_g_f_tot_KOM,
        "F_{gccKOM}": F_g_c_tot_KOM,
    }

    pp.write_lists_to_file("simulation_results_comparison.txt", lists_to_write)

    data_to_save = {
        "data_lists": [
            [F_g_f_tot_LT2, F_g_f_tot_KFR, F_g_f_tot_GEG, F_g_f_tot_GCG, F_g_f_tot_KOM],
            [F_g_c_tot_LT2, F_g_c_tot_KFR, F_g_c_tot_GEG, F_g_c_tot_GCG, F_g_c_tot_KOM],
            [F_e_f_LT2, F_e_f_KFR, F_e_f_GEG, F_e_f_GCG, F_e_f_KOM],
            [F_e_c_LT2, F_e_c_KFR, F_e_c_GEG, F_e_c_GCG, F_e_c_KOM],
        ],
        "titles": [
            "Growth in fiber direction",
            "Growth in crossfiber direction",
            "Stretch in fiber direction",
            "Stretch in crossfiber direction",
        ],
        "ylabels": ["Distance", "Distance", "Stretch ratio", "Stretch ratio"],
    }
    output.mkdir(parents=True, exist_ok=True)

    with open(output / "10p_stretch_crossfiber.json", "w") as f:
        json.dump(data_to_save, f)
    print("Data saved as 10p_stretch_crossfiber.json")

    # Load data from JSON file
    with open(output / "10p_stretch_crossfiber.json", "r") as f:
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
    fig.legend(
        lines,
        ["LT2", "KFR", "GEG", "GCG", "KOM"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        fontsize=18,
    )

    plt.subplots_adjust(bottom=0.1)

    # Save the figure
    plt.savefig(output / "10p_stretch_crossfiber.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Plot saved as 10p_stretch_crossfiber.png")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
