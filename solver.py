#!/usr/bin/python3

def greedy(state):
    current_state = state
    while len(current_state.options()) != 0:
        best_option, best_costs = -1, -1
        for i, option in enumerate(current_state.options()):
            nxt = current_state.simulate_step(option)
            costs = nxt.costs()
            if best_costs == -1 or costs <= best_costs:
                best_costs = costs
                best_option = nxt
        current_state = best_option
    return current_state

# seems to find a solution, but veeeeery slow. 
# maybe i did something wrong or state space is just too huge.
def dijkstra(state):
    states = set({state})
    last_best_state = state

    while len(states) != 0:
        best_state, best_costs = None, -1
        for nxt in states:
            c = nxt.costs()
            if best_costs == -1 or c < best_costs:
                best_costs = c
                best_state = nxt
        states.remove(best_state)
        options = best_state.options()
        if len(options) == 0:
            return best_state

        for option in options:
            nxt = best_state.simulate_step(option)
            states.add(nxt)

def random(state):
    # generate random schedule
    from random import choice    
    current_state = state
    options = current_state.options()
    while len(options) != 0:
        current_state = current_state.simulate_step( choice(options) )
        options = current_state.options()
    return current_state

strategies = { "dijkstra" : dijkstra, "greedy":greedy, "random":random}

def find_solution(state, strategy):
    current_state = strategies[strategy](state)
    while current_state is not None:
        yield current_state
        current_state = current_state.last_state








