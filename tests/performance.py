from firedrake import *
import matplotlib.pyplot as plt
import time
import numpy as np

av_runtimes = []
dof = []
for i in range(5):
    mesh = UnitSquareMesh(10,10)
    V = VectorFunctionSpace(mesh, "CG", 1)

for i in range(5, 13):
    runtimes = np.zeros(10)
    N = i**2
    print('calculating for {}'.format(N))
    for it in range(10):
        mesh = UnitSquareMesh(N, N)
        start = time.time()
        V = VectorFunctionSpace(mesh, "CG", 1)
        end = time.time()
        runtimes[it] = end-start

    av_runtimes.append(np.mean(runtimes))
    dof.append(N*N)

plt.plot(av_runtimes, dof, '--o', color='red')
plt.title('Building FunctionSpace Object', fontsize='16')
plt.ylabel('DoF', fontsize='14')
plt.xlabel('time to build [s]', fontsize='14')
plt.show()



