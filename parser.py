#!/usr/bin/python3

class Problem:
    def __init__(self, name, size, jobs):
        self.name = name
        self.nr_jobs, self.nr_machines  = size
        self.jobs = jobs

    def __repr__(self):
        return "name: {}; #machines: {}; #jobs: {}\n{}".format(self.name,
                self.nr_machines, self.nr_jobs, "\n".join([str(job) for job in self.jobs]))

class Step:
    def __init__(self, job, machine, runtime, earliest, next_step=None):
        self.machine = machine
        self.job = job
        self.runtime = runtime
        self.earliest = earliest
        self.next_step = None

    def __repr__(self):
        return "s({}, {}, {})".format(self.job, self.machine, self.runtime)

class Job:
    def __init__(self, steps):
        self.__steps_from_list(steps)

    def __steps_from_list(self, steps):
        self.first_step = steps[0]
        cur_step = self.first_step
        for s in steps[1:]:
            cur_step.next_step = s
            cur_step = s

    # enumeration of all steps
    def steps(self):
        step = self.first_step
        while step is not None:
            yield step
            step = step.next_step

    def __repr__(self):
        return "job (machine/runtime):  {}".format(" ".join([str(step) for step in self.steps()]))


def get_problems(filename):
    txt = open(filename).read()
    txt = txt.split("+++++++++++++++++++++++++++++")
    problems_serialized = txt[4::2]
    problems = []
    for i, problem_serialized in enumerate(problems_serialized):
        lines = problem_serialized.split("\n")
        if i == len(problems_serialized) - 1:
            lines = lines[:-1]
        # remove overhead
        lines = lines[1:-1]
        name = lines[0]
        size = [ eval(x) for x in lines[1].split(" ")[1:] ]
        # parse remaining lines
        jobs = []
        for j, l in enumerate(lines[2:]):
            l = [ eval(x) for x in l.strip().split() ]
            steps = []
            for i in range(len(l)//2):
                machine = l[i*2]
                runtime = l[i*2+1]
                steps.append( Step(j, machine, runtime, 0) )
            jobs.append(Job(steps))
        problem = Problem(name, size, jobs)
        problems.append(problem)
    return problems
