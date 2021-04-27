import os
import matplotlib.pyplot as plt
import time
import numpy as np

names = ['orig', 'OO','builder']
colors = ['orangered', 'rebeccapurple', 'darkgreen']
markers = ['--o', '--s', '--^']

num_iters = 100

for idx, name in enumerate(names):

    os.system('cp ../firedrake/{}_functionspace.py ../firedrake/functionspace.py'.format(name))
    os.system('make -C ../')
    import firedrake

    for i in range(50):
        mesh = firedrake.UnitSquareMesh(10,10)
        V = firedrake.VectorFunctionSpace(mesh, "CG", 1)
    av_runtimes = []
    dof = []

    for i in range(12, 20):
        runtimes = np.zeros(num_iters)
        N = i**2
        print('calculating for {}'.format(N))
        for it in range(num_iters):
            mesh = firedrake.UnitSquareMesh(N, N)
            start = time.time()
            V = firedrake.VectorFunctionSpace(mesh, "CG", 1)
            end = time.time()
            runtimes[it] = end-start

        av_runtimes.append(np.mean(runtimes))
        dof.append(N*N)
    av_runtimes = np.array(av_runtimes)
    if name == 'orig':
        orig_runtimes = av_runtimes
        label = 'functional (original)'
    elif name == 'OO':
        label = 'plain OO'
        OO_diff_runtimes = orig_runtimes - av_runtimes
    elif name == 'builder':
        label = 'builder pattern'
        builder_diff_runtimes = orig_runtimes - av_runtimes

    plt.plot(dof, av_runtimes, markers[idx], color=colors[idx], label=label)

    del firedrake

plt.suptitle('Runtime to Build FunctionSpace Object', fontsize='16')
plt.title('8365U, gcc-9', fontsize='10')
plt.xlabel('DoF', fontsize='14')
plt.ylabel('time [s]', fontsize='14')
plt.legend(fontsize='14', loc='upper left')
plt.savefig('runtime_performance.png')
plt.show()


plt.suptitle('Runtime Difference to Build Functionspace Object', fontsize='16')
plt.title('8365U, gcc-9', fontsize='10')
plt.xlabel('DoF', fontsize='14')
plt.ylabel('time difference [s] (functional - OO)', fontsize='14')
plt.plot(dof, OO_diff_runtimes, markers[1], color=colors[1], label='plain OO')
plt.plot(dof, builder_diff_runtimes, markers[2], color=colors[2], label='builder pattern')
plt.legend(fontsize='14', loc='upper left')
plt.savefig('runtime_difference.png')
plt.show()




