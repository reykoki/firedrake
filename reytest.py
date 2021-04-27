from firedrake import *
n = 30
mesh = UnitSquareMesh(n, n)

V = VectorFunctionSpace(mesh, "CG", 2)
V_out = VectorFunctionSpace(mesh, "CG", 1)
