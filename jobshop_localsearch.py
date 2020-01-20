#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import math
import random as rand
import operator as op
from dataclasses import dataclass
from functools import reduce

from typing import List, Dict, Set
# %%

Job = int
StepId = int
Machine = int
Duration = int

# %%

@dataclass
class Step:
    """ A step is a mere wrapper to capture all of its relevant data. """

    job: Job
    step_id: StepId
    machine: Machine
    duration: Duration

    def __eq__(self, other):
        return self.job == other.job and self.step_id == other.step_id

    def __hash__(self):
        return hash((self.job, self.step_id))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "S({j}, {s}, m={m}, d={d})".format(j=self.job, s=self.step_id, d=self.duration, m=self.machine)

# %%

Chronology = Dict[Machine, List[Step]]
Jobs = List[List[Step]]

# %%

def generate_steps(job: Job, n_steps: int) -> List[Step]:
    """ Creates steps for random machines and random duration for the *job*. """

    return [Step(job, i, rand.randint(0, n_machines-1), rand.randint(1, max_step_duration)) for i in range(n_steps)]

def gather_steps(machine: Machine, jobs: List[List[Step]]) -> List[Step]:
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

    pass

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

    def __init__(self,
                 jobs: List[List[Step]],
                 chronology: Dict[Machine, List[Step]],
                 dependency_graph: Dict[Step, List[Step]]=None,
                 dependant_steps: Dict[Step, List[Step]]=None):

        self.jobs = {i: jobs[i] for i in range(len(jobs))}
        self.chronology = chronology

        all_steps = reduce(op.add, jobs)

        if dependency_graph is None:
            self.dependency_graph: Dict[Step, List[Step]] = {}
            self.dependant_steps: Dict[Step, List[Step]] = {}
            for step in all_steps:
                step_predecessors = set()

                machine_seq = chronology[step.machine]
                machine_pred_idx = machine_seq.index(step) - 1
                if machine_pred_idx >= 0:
                    step_predecessors.add(machine_seq[machine_pred_idx])

                job_pred_idx = step.step_id - 1
                if job_pred_idx >= 0:
                    step_predecessors.add(jobs[step.job][job_pred_idx])

                self.dependency_graph[step] = step_predecessors

                for predecessor in step_predecessors:
                    dependants = self.dependant_steps.setdefault(predecessor, set())
                    dependants.add(step)
        else:
            self.dependency_graph = dependency_graph
            self.dependant_steps = dependant_steps

        self.n_steps = len(all_steps)

    def mark_step_done(self, step: Step) -> OperationDependencies:
        """
            Removes a step as dependency of all other steps.

            Thus, after completing one executable step, other steps might
            become executable in turn.

            This raises an `UnsatisfiedDepdenciesError` if the `step` still has
            dependencies left.
        """

        if self.dependency_graph[step]:
            raise UnsatisfiedDepdenciesError()

        new_graph = copy.deepcopy(self.dependency_graph)
        new_dependants = copy.deepcopy(self.dependant_steps)

        if step in new_dependants:
            for dependant in new_dependants[step]:
                new_graph[dependant].remove(step) # no step depends on this step anymore
            del new_dependants[step] # this step is no dependency anymore

        del new_graph[step]

        return OperationDependencies(list(self.jobs.values()), self.chronology, new_graph, new_dependants)

    def get_executable_steps(self) -> List[Step]:
        """
            Queries for all steps which currently have on dependencies and
            therefore are ready for execution.
        """

        return list(
                map(lambda s: s[0], # unwrap the steps
                    filter(lambda s: s[1] == 0, # get all steps with 0 depedencies
                           # wrap the steps with their dependency counter
                           map(lambda s : (s[0], len(s[1])),
                               self.dependency_graph.items()))))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.dependency_graph)

# %%


# %%

class Schedule:
    """ A schedule describes which step is started at which point in time. """

    def __init__(self, jobs: List[List[Step]], chronology: Dict[Machine, List[Step]]):
        self.jobs = jobs
        self.chronology = chronology
        self.step_execution_time: Dict[Step, int] = {}
        self.schedule: Dict[int, Dict[int, Step]] = {}
        self.n_steps = 0

        for machine in chronology:
            self.schedule[machine] = {}

    def assign(self, steps):
        """
            Starts each of the steps at the soonest time possible.

            This assumes that all constraint regarding step execution are
            satisfied.
        """

        for step in steps:
            machine_ready_time, predecessor_done_time = None, None

            if len(self.schedule[step.machine]) == 0: # first step on machine
                machine_ready_time = 0
            else:
                last_step_execution = max(self.schedule[step.machine].keys())
                last_step_on_machine = self.schedule[step.machine][last_step_execution]
                machine_ready_time = self.step_execution_time[last_step_on_machine] + last_step_on_machine.duration

            if step.step_id == 0: # first step in job
                predecessor_done_time = 0
            else:
                predecessor_step = self.jobs[step.job][:step.step_id][-1]
                predecessor_done_time = self.step_execution_time[predecessor_step] + predecessor_step.duration

            step_execution_time = max(machine_ready_time, predecessor_done_time)

            self.step_execution_time[step] = step_execution_time
            self.schedule[step.machine][step_execution_time] = step

        self.n_steps += len(steps)

    def duration(self):
        return max(map(self._end_time(), range(len(self.schedule))))

    def _end_time(self):
        schedule = self
        def do_calc(machine):
            last_start = max(schedule.schedule[machine])
            return last_start + schedule.schedule[machine][last_start].duration
        return do_calc

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.schedule)

# %%

def chronology2schedule(jobs: Jobs, chronolgy: Chronology) -> Schedule:
    """ Converts a chronology to the quickest schedule induced by it. """

    dependencies = OperationDependencies(jobs, chronolgy)
    schedule = Schedule(jobs, chronolgy)
    
    i = 0
    executable_steps = dependencies.get_executable_steps()
    while executable_steps:
        schedule.assign(executable_steps)
        for step in executable_steps:
            dependencies = dependencies.mark_step_done(step)
        executable_steps = dependencies.get_executable_steps()
        i += 1

    if schedule.n_steps < dependencies.n_steps:
        raise CyclicDependencyError()

    return schedule

# %%

def find_neighbors(jobs, chronology) -> List[(Chronology, Schedule)]:
    allowed_chronologies = []
    for machine, steps in chronology.items():
        swaps = all_2_swaps(steps)
        for swap in swaps:
            try:
                resulting_chrono = copy.deepcopy(chronology)
                resulting_chrono[machine] = swap
                resulting_schedule = chronology2schedule(jobs, chronology)
                allowed_chronologies.append((resulting_chrono, resulting_schedule))
            except CyclicDependencyError:
                pass

    return allowed_chronologies

# %%

def random_chronology(jobs) -> Chronology:
    # get initial steps
    choices = {m: [step for step in gather_steps(m,jobs) if step.step_id == 0] for m in range(n_machines)}
    # start w/ empty chronology
    chronology = {}
    # contains steps which are already in the chronology
    # while we have steps left ...
    while any([len(choices[machine])>0 for machine in choices]):
        # add one random step for each machine to the chronology
        for machine in choices:
            if machine not in chronology:
                chronology[machine] = []
            if len(choices[machine]) > 0:
                choosen = rand.choice(choices[machine])
                chronology[machine].append(choosen)
                # remove the choosen step from the set of choices, but before that: 
                # add its successor to the list of possible choices
                if choosen.step_id + 1 < len(jobs[choosen.job]):
                    next_step = jobs[choosen.job][choosen.step_id + 1]
                    choices[next_step.machine].append(next_step)
                choices[machine].remove(choosen)
    return chronology

# %%

def search_hillclimber_iterated(jobs, n_iterations=10):
    steps = []
    current_chronology = random_chronology(jobs)
    best_schedule = chronology2schedule(jobs, current_chronology)
    i = 0
    while i < n_iterations:
        s = 0
        print(i)
        current_chronology = random_chronology(jobs)
        plateaued = False
        while not plateaued:
            neighbors = find_neighbors(jobs, current_chronology)
            neighbors_eval = { neighbor[1].duration() : neighbor[0] for neighbor in neighbors }
            best_eval = min(neighbors_eval)
            best_neighbor = neighbors_eval[best_eval]
            if best_eval < chronology2schedule(jobs, current_chronology).duration():
                current_chronology = best_neighbor
            else:
                plateaued = True
            s += 1
        steps.append(s)
        i += 1
        current_schedule = chronology2schedule(jobs, current_chronology)
        if current_schedule.duration() < best_schedule.duration():
            best_schedule = current_schedule

    return best_schedule, steps

# %%

def shc_selection_prop(current_eval, new_eval, T=10):
    return 1 / (1 + math.e**((current_eval - new_eval) / T))

def search_hillclimber_stochastic(jobs, n_iterations=100, merit_weight=10):
    i = 0
    while i < n_iterations:
        current_chronology = random_chronology(jobs)
        print(i)
        neighbors = find_neighbors(jobs, current_chronology)
        selected_neighbor = rand.choice(neighbors)
        selection_eval = selected_neighbor[1].duration()
        current_eval = chronology2schedule(jobs, current_chronology).duration()

        choice_prop = shc_selection_prop(current_eval, selection_eval, merit_weight)
        update = rand.uniform(0, 1) < choice_prop

        if update:
            current_chronology = selected_neighbor[0]

        i += 1
    return chronology2schedule(jobs, current_chronology)

# %%

def sa_selection_prop(current_eval, new_eval, T):
    return math.e**((-current_eval - new_eval) / T)

def cool_down(current_chronology, iterations, temp, temp_max, cooling_ratio):
    return temp_max * math.e**(-iterations * cooling_ratio)

def search_simulatedannealing(jobs, n_iterations=100, initial_temp=100, min_temp=0.5, max_temp=100, cooling_ratio=.5):
    i = 0
    best_schedule = None
    while i < n_iterations:
        current_chronology = random_chronology(jobs)
        current_schedule = chronology2schedule(jobs, current_chronology)

        if best_schedule is None:
            best_schedule = current_schedule

        print("## Run", i)
        temp = initial_temp
        r = 0
        while temp > min_temp:
            print("temp={}".format(temp))
            neighbors = find_neighbors(jobs, current_chronology)
            selected_neighbor = rand.choice(neighbors)
            selected_schedule = selected_neighbor[1]
            selection_eval = selected_schedule.duration()
            current_eval = current_schedule.duration()

            if selection_eval < current_eval:
                current_chronology = selected_neighbor[0]
                current_schedule = selected_schedule
            elif rand.uniform(0, 1) < sa_selection_prop(current_eval, selection_eval, temp):
                current_chronology = selected_neighbor[0]
                current_schedule = selected_schedule
                temp = cool_down(current_chronology, r, temp, max_temp, cooling_ratio)
                r += 1
            else:
                temp = cool_down(current_chronology, r, temp, max_temp, cooling_ratio)
                r += 1

        i += 1

        if current_schedule.duration() < best_schedule.duration():
            best_schedule = current_schedule
            print("New best:", best_schedule.duration())


    return best_schedule

n_machines = -1

def set_params(problem):
    global n_machines
    n_machines = problem.nr_machines

# %%

if __name__=="__main__":
    max_step_duration = 20

    n_machines = 5 #rand.randint(2, 20)
    n_jobs = 5 #rand.randint(1, 20)
    n_steps_per_job = 5 #rand.randint(1, 20)

    jobs = [generate_steps(job, n_steps_per_job) for job in range(n_jobs)]
    print(jobs)
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

    schedule = chronology2schedule(jobs, chronology)
    print("schedule:")
    print(schedule)
    from renderer import render_jls
    render_jls(n_machines,n_jobs,schedule)

