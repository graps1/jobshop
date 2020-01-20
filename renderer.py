#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import jobshop_localsearch as jls

fontsize=14

def __display(mat):
    fig = plt.figure(figsize=(20,7))
    ax = fig.add_subplot(111)
    plt.xlabel("time", size=fontsize)
    plt.ylabel("machines", size=fontsize)
    ax.imshow(mat, interpolation ="nearest")
    ax.set_aspect(aspect="auto")
    plt.show()

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
    __display(mat)

def render_improvement(values):
    fig = plt.figure(figsize=(20,7))
    ax = fig.add_subplot(111)
    plt.xlabel("iteration", size=fontsize)
    plt.ylabel("evaluation", size=fontsize)
    plt.plot(range(len(values)), values, "gx:")
    plt.show()

def render_jls(nr_machines, nr_jobs, schedule : jls.Schedule):
    s = schedule.schedule
    
    # find maximal runtime
    maxrt = 0   
    for m,dic in s.items():
        for t,stp in dic.items():
            maxrt = max(maxrt, t+stp.duration)
    mat = np.zeros((nr_machines, maxrt))
    
    for m,dic in s.items():
        for t,stp in dic.items():
            mat[m, t:(t+stp.duration)] = (stp.job+0.5) / nr_jobs
    __display(mat)
