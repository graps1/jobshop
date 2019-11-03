#!/usr/bin/python3

from state import State
from parser import get_problems
from solver import find_solution
from renderer import render

print("reading jobshop description...",end=" ")
# creates a list of problems from the files content.
problems = get_problems("jobshop1.txt")
print("done.")

# select the fifth problem
problem = problems[5]
# strategy is a string
strategy = ["dijkstra", "greedy", "random"][1]

print(problem)
print("."*100)

# initial steps yields the steps the scheduler can take in the beginning
initial_steps = [job.first_step for job in problem.jobs]
state = State(problem.nr_machines, initial_steps)

print("solving...",end="")
# the trajectory is a schedule
trajectory = list(find_solution(state, strategy))[::-1]
print("done.")


for el in trajectory:
   print(el)

render(problem, trajectory)
