#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import random as rand
import itertools as it
import operator as op
from functools import reduce

from typing import List, Dict, Set
# %%

@dataclass
class Step:
    """ A step is a mere wrapper to capture all of its relevant data. """
    
    job: int
    step_id: int
    machine: int
    duration: int
    
    def __str__(self):
        return "S({j}, {s}): {d}@{m}".format(j=self.job, s=self.step_id, d=self.duration, m=self.machine)

# %%

def generate_steps(job: int, n_steps: int) -> List[Step]:
    """ Creates steps for random machines and random duration for the *job*. """
    
    return [Step(job, i, rand.randint(0, n_machines-1), rand.randint(1, max_step_duration)) for i in range(n_steps)]

def gather_steps(machine: int, jobs: List[Step]) -> List[Step]:
    """ Collects all of the steps in *jobs* that should be run on *machine*."""
    
    return list(filter(lambda s: s.machine == machine, [step for steps in jobs for step in steps]))

def all_2_swaps(steps: List[Step]) -> List[List[Step]]:
    """
        Calculates all possible sequences of steps that may be created by
        swapping two steps of the given sequence.
    """
    
    swaps = []
    for first_idx, step1 in enumerate(steps):
        for second_idx, step2 in enumerate(steps[first_idx:]):
            if first_idx == first_idx + second_idx:
                continue
            swapped = list(steps)
            swapped[first_idx], swapped[first_idx+second_idx] = swapped[first_idx+second_idx], swapped[first_idx]
            swaps.append(swapped)
    return swaps

# %%

class CyclicDependencyError(Exception):
    """
        Exception to indicate that a set of dependencies is cyclic and may
        therefore not be fulfilled.
    """
    
    def __init__(self, step):
        self.step = step
        
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        "Cyclic dependency detected at step {}".format(self.step)

class UnsatisfiedDepdenciesError(Exception):
    """
        Exception to indiciate that some step still has dependencies left
        unsatisfied and may therefore not be executed.
    """
    
    pass

class OperationDependencies:
    """
        Operation depdencies describe which steps need to be executed prior
        to other steps.
        
        It's main data structure it the `step_dependecies` dict which maps
        each step to a list of other steps that need to be run before.
        These dependecies may be modified by marking suitable steps as executed
        potentially resulting in other steps becoming executable.
        
        `OperationDependencies` should be treated as immutable. Therefore each
        modifying operation instead returns a new instance.
    """
    
    def __init__(self, jobs: List[List[Step]], chronoloy: Dict[int, List[Step]], chronology_dependencies = {}, step_dependencies = {}):
        self.jobs = {i: jobs[i] for i in range(len(jobs))}
        self.chronology = chronology
        self.chronology_dependencies = chronology_dependencies
        self.step_dependencies = step_dependencies
        
        # if the dependencies have not been initialized yet, do so
        if not chronology_dependencies:
            for step in reduce(op.add, jobs):
                self.step_dependencies[step] = {}
                self.chronology_dependencies[step] = []
            
            # gather dependencies due to operation sequence of the machines
            for machine in chronology:
                for step_idx, step in enumerate(chronology[machine]):
                    if step_idx == 0:
                        continue
                    self.chronology_dependencies[step] += chronology[machine][:step_idx]
            
            # gather dependecies due to prior steps in the job as well as
            # transitive depedencies
            for job in jobs:
                for step_idx, step in enumerate(job):
                    self.step_dependencies[step] = set(self.chronology_dependencies[step] + self._collect_transitive_dependencies(step))
        else:
            self.chronology_dependencies = copy.deepcopy(chronology_dependencies)
            self.step_dependencies = copy.deepcopy(step_dependencies.copy())
                
    def mark_step_done(self, step: Step) -> OperationDependencies:
        """
            Removes a step as dependency of all other steps.
        
            Thus, after completing one executable step, other steps might
            become executable in turn.
            
            This raises an `UnsatisfiedDepdenciesError` if the `step` still has
            dependencies left.
        """
        
        if self.step_dependencies[step]:
            raise UnsatisfiedDepdenciesError()
            
        resulting_ods = self.__deepcopy__()
        
        # remove the step itself
        resulting_ods.step_dependencies.pop(step)
        
        # as well as all of its occurences as a dependency of aother steps
        for other_deps in resulting_ods.step_dependencies.values():
            if step in other_deps:
                other_deps.remove(step)
        
        return resulting_ods
    
    def get_executable_steps(self) -> List[Step]:
        """
            Queries for all steps which currently have on dependencies and
            therefore are ready for execution.
        """
        
        return list(
                map(lambda s: s[0], # unwrap the steps
                    filter(lambda s: s[1] == 0, # get all steps with 0 depedencies
                           # wrap the steps with their dependency counter
                           map(lambda s : (s, len(self.step_dependencies[s])),
                               self.step_dependencies))))
        
    def _collect_transitive_dependencies(self, step: Step, visited: Set[Step] = set()) -> List[Step]:
        """
            Determines the transitive closure of the depends-on relation for step.
            
            In the resulting list, some steps may appear multiple times in case
            more than one other step depends on it.
        """
        
        # The current implementation is pretty straightforward but heavily
        # ineffective.
        
        if step in visited:
            raise CyclicDependencyError(step)
        
        visited = visited.copy()
        visited.add(step)
        
        # predecessors are those steps for the current job, that have to be
        # executed before this step
        predecessors = self.jobs[step.job][:step.step_id]        
        chronology_dependencies = self.chronology_dependencies[step]        
        direct_dependencies = predecessors + chronology_dependencies
        
        transitive_dependencies = direct_dependencies.copy()
        
        for dependency in direct_dependencies:
            transitive_dependencies += self._collect_transitive_dependencies(dependency, visited)
            
        return transitive_dependencies
        
    def __deepcopy__(self) -> OperationDependencies:
        return OperationDependencies(list(self.jobs.values()), self.chronology, self.chronology_dependencies, self.step_dependencies)
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return str(self.step_dependencies)
    
# %%

class Schedule:
    """ A schedule describes which step is started at which point in time. """
    
    def __init__(self, jobs: List[List[Step]], chronology: Dict[int, List[Step]]):
        self.jobs = jobs
        self.chronology = chronology
        self.step_execution_time: Dict[Step, int] = {}
        self.schedule: Dict[int, Dict[int, Step]] = {}
        
        for machine in chronology:
            self.schedule[machine] = {}
        
    def assign(self, *steps: Step):
        """
            Starts each of the steps at the soonest time possible.
            
            This assumes that all constraint regarding step execution are
            satisfied.
        """
        
        for step in steps:
            last_step_on_machine = max(self.schedule[step.machine].keys())
            predecessor_step = self.jobs[step.job][:step.step_id][-2]
            
            machine_ready_time = self.step_execution_time[last_step_on_machine] + last_step_on_machine.duration
            predecessor_done_time = self.step_execution_time[predecessor_step] + predecessor_step.duration
            
            step_execution_time = max(machine_ready_time, predecessor_done_time) + 1
            
            self.step_execution_time[step] = step_execution_time
            self.schedule[step.machine][step_execution_time] = step
        


# %%

max_step_duration = 20

n_machines = 2 #rand.randint(2, 20)
n_jobs = 2 #rand.randint(1, 20)
n_steps_per_job = 3 #rand.randint(1, 20)

jobs = [generate_steps(job, n_steps_per_job) for job in range(n_jobs)]

""" 
    A chronology describes the order in which machines execute steps.
    It differs from a complete schedule in that it does not contain any
    timing information. Instead, it only provides a successor relation meaning
    that a certain step has to be executed before another.
    
    Starting with a chronology it is quite easy to create a full-fledged
    schedule: a padding has to be introduced to accomodate for the dependencies
    between steps of the same job.
    In one special case a chronology may be degenerated meaning that it is
    impossible to to create a schedule for it. This case occurs if the
    chronology contains a cyclic dependency.
"""
chronology = {m: gather_steps(m, jobs) for m in range(n_machines)}

print("machines:", n_machines)
print("jobs:", n_jobs)
print("steps / job:", n_steps_per_job)
print("jobs:", jobs)
print("chronology:", chronology)

# neighbors = reduce(lambda m1, m2: it.product(m1, m2), map(it.permutations, chronology.values()))

# %%

neighbors = {}
for machine, steps in chronology.items():
    neighbors[machine] = all_2_swaps(steps)

n_neighbors = reduce(op.add, map(len, neighbors.values()))

ods = OperationDependencies(jobs, chronology)
