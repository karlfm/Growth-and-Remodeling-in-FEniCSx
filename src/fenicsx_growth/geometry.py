import dolfinx
from mpi4py import MPI
import numpy as np
from dolfinx.io import XDMFFile

from . import ddf


def create_unit_cube(X):
    xs = X[0]
    ys = X[1]
    zs = X[2]
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, len(xs), len(ys), len(zs))


def create_unit_square(X):
    xs = X[0]
    ys = X[1]
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, len(xs), len(ys))


def create_box(X: list[list[float]]):
    xs = X[0]
    ys = X[1]
    zs = X[2]
    return dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([xs[0], ys[0], zs[0]]), np.array([xs[-1], ys[-1], zs[-1]])],
        [len(xs), len(ys), len(zs)],
        cell_type=dolfinx.mesh.CellType.hexahedron,
    )


def get_boundary_nodes(mesh, X):
    output = []
    dimension = mesh.topology.dim
    boundary_dimension = dimension - 1

    for i, x in enumerate(X):
        x_min_edge = x[0]
        x_max_edge = x[-1]

        x_min = dolfinx.mesh.locate_entities(
            mesh, boundary_dimension, lambda x: np.isclose(x[i], x_min_edge)
        )
        x_max = dolfinx.mesh.locate_entities(
            mesh, boundary_dimension, lambda x: np.isclose(x[i], x_max_edge)
        )

        output.append((x_min, x_max))

    return output


def main():
    mesh = create_unit_cube(1, 1, 1)
    boundary_conditions = [(1, 1), (2, 2), (3, 3)]
    facet_tags = ddf.set_boundary_types(mesh, boundary_conditions)

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    with XDMFFile(mesh.comm, "facet_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tags, mesh.geometry)

    breakpoint()


if __name__ == "__main__":
    main()
