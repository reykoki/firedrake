from firedrake import *
import pytest
from petsc4py import PETSc
from petsc4py.PETSc import ViewerHDF5
from pyop2.mpi import COMM_WORLD
from pyop2 import RW
import os


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('family_degree', [("CG", 2),
                                           ("CG", 3)])
@pytest.mark.parametrize('format', [ViewerHDF5.Format.HDF5_PETSC, ])
def test_io_function_simplex(family_degree, format, tmpdir):
    # Parameters
    family, degree = family_degree
    filename = os.path.join(str(tmpdir), "test_io_function_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    ntimes = 3
    fs_name = "example_function_space"
    func_name = "example_function"
    # Initially, load an existing triangular mesh.
    comm = COMM_WORLD
    meshA = Mesh("./docs/notebooks/stokes-control.msh", comm=comm)
    #meshA = UnitSquareMesh(2, 1)
    meshA.init()
    plexA = meshA.topology.topology_dm
    VA = FunctionSpace(meshA, family, degree, name=fs_name)
    x, y = SpatialCoordinate(meshA)
    fA = Function(VA, name=func_name)
    fA.interpolate(x * y * y)
    meshA.save(filename, format=format)
    VA.save(filename)
    fA.save(filename)
    volA = assemble(fA * x * y * dx)
    # Load -> View cycle
    grank = COMM_WORLD.rank
    for i in range(ntimes):
        mycolor = (grank > ntimes - i)
        comm = COMM_WORLD.Split(color=mycolor, key=grank)
        if mycolor == 0:
            # Load
            meshB = Mesh(filename, comm=comm)
            VB = FunctionSpace(meshB, name=fs_name, filename=filename)
            fB = Function(VB, name=func_name, filename=filename)
            x, y = SpatialCoordinate(meshB)
            volB = assemble(fB * x * y * dx)
            # Check
            print("i = ", i)
            print("volA = ", volA)
            print("volB = ", volB)
            assert abs(volB - volA) < 1.e-7
            # Save
            meshB.save(filename, format=format)
            VB.save(filename)
            fB.save(filename)


if __name__ == "__main__":
    test_io_function_simplex(("CG", 3), ViewerHDF5.Format.HDF5_PETSC, "./")
