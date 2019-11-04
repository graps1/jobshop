#!/usr/bin/python3

from state import State
from parser import get_problems
from solver import find_solution
from renderer import render

import argparse
strategies = ["greedy", "dijkstra", "random"]
argparser = argparse.ArgumentParser("main")
argparser.add_argument("-f", "--file", 
        help="the file the problems should be read from.",
        default="jobshop1.txt",
        type=str)
argparser.add_argument("-p", "--problem", 
        type=int,
        help="the number of the problem which should be solved.",
        default=1)
argparser.add_argument("-s", "--strategy", 
        help="the strategy which should be choosen.", 
        choices=strategies,
        default=strategies[0])
args = argparser.parse_args()

print("reading jobshop description...",end=" ")
# creates a list of problems from the files content.
problems = get_problems(args.file)
print("done.")

# select the fifth problem
problem = problems[int(args.problem)]
# strategy is a string

print(problem)
print("."*100)

# initial steps yields the steps the scheduler can take in the beginning
initial_steps = [job.first_step for job in problem.jobs]
state = State(problem.nr_machines, initial_steps)

print("solving...",end="")
# the trajectory is a schedule
trajectory = list(find_solution(state, args.strategy))[::-1]
print("done.")


for el in trajectory:
   print(el)

render(problem, trajectory)
