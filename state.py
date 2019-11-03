#!/usr/bin/python3
from functools import reduce

# state stores the available options
# and the current state of the machines
class State:
    def __init__(self, nr_machines, initial_steps=None, last_state=None, last_step=None):
        self.nr_machines = nr_machines
        self.last_state = last_state
        self.last_step = last_step
        self.mapping = {}
        for i in range(nr_machines):
            self.mapping[i] = {"options":[], "time":0}
        if initial_steps is not None:
            for step in initial_steps:
                self.mapping[step.machine]["options"].append(step)

    def get_options(self):
        return self.mapping

    def options(self):
        options = [opt[1]["options"] for opt in self.mapping.items()]
        options = reduce(lambda x,y: x+y, options)
        return options

    def costs(self):
        overall = 0
        m = max([item[1]["time"] for item in self.mapping.items()])
        for item in self.mapping.items():
            overall += m - item[1]["time"]
        return overall

    def simulate_step(self, step):
        state = State(self.nr_machines)
        for i in range(self.nr_machines):
            state.mapping[i]["options"] = self.mapping[i]["options"][:]
            state.mapping[i]["time"] = self.mapping[i]["time"]
        state.mapping[step.machine]["options"].remove(step)
        state.mapping[step.machine]["time"] = max(state.mapping[step.machine]["time"],
                step.earliest) + step.runtime

        state.last_state = self
        state.last_step = step

        if step.next_step is not None:
            step.next_step.earliest = state.mapping[step.machine]["time"]
            state.mapping[step.next_step.machine]["options"].append(step.next_step)

        return state

    def __repr__(self):
        return "-"*100 + "\n" + "\n".join(["{}\t @ {}\t: {}".format(key,
            self.mapping[key]["time"], self.mapping[key]["options"]) for key in self.mapping])


