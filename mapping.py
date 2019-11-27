#!/usr/bin/local/python3


import jobshop_localsearch as jls
import parser as ps
import state as st
from typing import List


def map_step_to_jls(id,step : ps.Step) -> jls.Step:
    return jls.Step(step.job, id, step.machine, step.runtime)

def map_job_to_jls(job : ps.Job) -> List[jls.Step]:
    ret = []
    for i,stp in enumerate(job.steps()):
        ret.append(map_step_to_jls(i,stp))
    return ret

def map_problem_to_jls(problem : ps.Problem) -> List[List[jls.Step]]:
    ret = []
    for job in problem.jobs:
        ret.append(map_job_to_jls(job))
    return ret
    
    
