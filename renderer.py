#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

def render(problem, trajectory):
    runtimes = [ opt[1]["time"] for opt in trajectory[-1].get_options().items() ]
    mat = np.zeros((problem.nr_machines, max(runtimes)))

    last_state = trajectory[0]
    for state in trajectory[1:]:
        for i in range(problem.nr_machines):
            if state.last_step is not None:
                rt_off = max(last_state.get_options()[i]["time"], state.last_step.earliest)
                rt_stop = state.get_options()[i]["time"]
                mat[i, rt_off:rt_stop] = (state.last_step.job+0.5) / problem.nr_jobs
        last_state = state

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(mat, interpolation ="nearest")
    ax.set_aspect(aspect="auto")
    plt.show()


