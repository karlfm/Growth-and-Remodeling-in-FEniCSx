import dolfinx
from pathlib import Path
import basix
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def write_scalar_to_paraview(filename, mesh, scalars):
    function_space = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))  # , (1,  )
    function = dolfinx.fem.Function(function_space)

    filename = Path(filename)
    filename.unlink(missing_ok=True)
    filename.with_suffix(".h5").unlink(missing_ok=True)
    foutStress = dolfinx.io.XDMFFile(mesh.comm, filename, "w")
    foutStress.write_mesh(mesh)

    for i, scalar in enumerate(scalars):
        stress_expr = dolfinx.fem.Expression(
            scalar, function_space.element.interpolation_points()
        )
        function.interpolate(stress_expr)
        foutStress.write_function(function, i)


def write_vector_to_paraview(filename, mesh, us):
    vector_element = basix.ufl.element(
        family="Lagrange", cell=str(mesh.ufl_cell()), degree=1, shape=(3,)
    )
    function_space = dolfinx.fem.functionspace(mesh, vector_element)
    function = dolfinx.fem.Function(function_space)

    filename = Path(filename)
    filename.unlink(missing_ok=True)
    filename.with_suffix(".h5").unlink(missing_ok=True)
    fout = dolfinx.io.XDMFFile(mesh.comm, filename, "w")
    fout.write_mesh(mesh)

    for i, u in enumerate(us):
        function.interpolate(u)
        fout.write_function(function, i)


def write_vector_to_paraview2(filename, mesh, us):
    if len(us) == 0:
        raise RuntimeError("No vectors to write")

    u = us[0].copy()
    with dolfinx.io.VTXWriter(
        mesh.comm, Path(filename).with_suffix(".bp"), [u], engine="BP4"
    ) as vtx:
        for i, ui in enumerate(us):
            u.x.array[:] = ui.x.array
            vtx.write(float(i))


def write_tensor_to_paraview(filename, mesh, tensors):
    quadrature_element = basix.ufl.quadrature_element(
        scheme="default",
        degree=8,
        value_shape=(3, 3),
        cell=basix.CellType[mesh.ufl_cell().cellname()],
    )

    tensor_element = basix.ufl.element(
        family="DG", cell=str(mesh.ufl_cell()), degree=0, shape=(3, 3)
    )
    tensor_space = dolfinx.fem.functionspace(mesh, tensor_element)
    quadrature_space = dolfinx.fem.functionspace(mesh, quadrature_element)
    function = dolfinx.fem.Function(tensor_space)

    filename = Path(filename)
    filename.unlink(missing_ok=True)
    filename.with_suffix(".h5").unlink(missing_ok=True)
    fout = dolfinx.io.XDMFFile(mesh.comm, filename, "w")
    fout.write_mesh(mesh)

    for i, tensor in enumerate(tensors):
        # tensor_as_quadrature_expression = dolfinx.fem.Expression(tensor, quadrature_space.element.interpolation_points())
        # tensor_as_tensor_expression = dolfinx.fem.Expression(tensor, quadrature_space.element.interpolation_points())
        function.interpolate(tensor)
        fout.write_function(function, i)


def format_number(num):
    return f"{num:.15f}".rstrip("0").rstrip(".")  # Adjust 15 to your desired precision


def write_lists_to_file(filename, lists_dict):
    with open(filename, "w") as f:
        for name, lst in lists_dict.items():
            formatted_list = [
                format_number(x) if isinstance(x, float) else str(x) for x in lst
            ]
            f.write(f"{name} = [{', '.join(formatted_list)}]\n\n")


def make_plot(fig, gs, pos, data, title, ylabel, time):
    ax = fig.add_subplot(gs[pos])
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    x = np.linspace(0, 1, len(data[0][:]))
    for i, d in enumerate(data):
        ax.plot(time, d[:], label=f"Data {i+1}", linewidth=2.5)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.grid(True)
    return ax


def main():
    # Load data from JSON file
    with open("../GrowthLaws/10p_stretch_fiber.json", "r") as f:
        loaded_data = json.load(f)

    data_lists = loaded_data["data_lists"]
    titles = loaded_data["titles"]
    ylabels = loaded_data["ylabels"]

    # Create figure with GridSpec
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(2, 2)
    axes = []
    time = np.linspace(0, 200, 2000 + 1)  # number of days measured in N steps
    for i in range(4):
        ax = make_plot(fig, gs, i, data_lists[i], titles[i], ylabels[i], time)
        # if i == 0:
        #     ax.set_ylim(0.85, 1.01)  # Set y-axis limits
        # elif i == 1:
        #     ax.set_ylim(0.85, 1.2)
        # elif i == 2:
        #     ax.set_ylim(0.85, 1.05)
        # elif i == 3:
        #     ax.set_ylim(0.98, 1.5)
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
    plt.savefig("10p_stretch_fiber.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Plot saved as 10p_stretch_fiber.png")


if __name__ == "__main__":
    main()
