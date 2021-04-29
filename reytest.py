from firedrake import *
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8), dpi=80)
n = 30
mesh = UnitSquareMesh(n, n)
fig.axes = plt.subplots()
triplot(mesh, axes=axes)


