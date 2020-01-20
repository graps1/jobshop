#!/usr/bin/python3

from parser import get_problems

import argparse
argparser = argparse.ArgumentParser("main")

argparser.add_argument("-f", "--file", 
        help="the file the problems should be read from.",
        default="jobshop1.txt",
        type=str)
argparser.add_argument("-p", "--problem", 
        type=int,
        help="the number of the problem which should be solved.",
        default=1)

#strategies = ["greedy", "dijkstra", "random"]
#argparser.add_argument("-s", "--strategy", 
#        help="the strategy which should be choosen.", 
#        choices=strategies,
#        default=strategies[0])
args = argparser.parse_args()

print("reading jobshop description...",end=" ")

problems = get_problems(args.file)
print("done.")
problem = problems[int(args.problem)]
print(problem)
print("."*100)

# map problem to local-search data structure
import jobshop_localsearch as jls
from mapping import map_problem_to_jls
from renderer import render_jls
jobs = map_problem_to_jls(problem)
jls.set_params(problem)
schedule,_ = jls.search_hillclimber_iterated(jobs)
render_jls(problem.nr_machines, problem.nr_jobs, schedule)

