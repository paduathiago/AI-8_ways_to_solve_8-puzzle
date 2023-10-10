# 8 ways to solve 8-puzzle - Introduction to Artificial Intelligence
This project implements and compairs different seach methods applied to the 8-puzzle problem.

## The following Algorithms have been implemented to solve the problem:

### Uninformed Search
- Breadth-first search
- Iterative deepening search
- Uniform-cost search

### Informed Search
- A* search
- Greedy best-first search

Each informed search algorithm uses the following heuristics:
- Number of misplaced tiles
- Manhattan distance

### Local Search
- Hill Climbing

## How to run the program
The program can be run from the command line using the following command:
```
python3 tp1.py <method> <board> <PRINT> (optional)
```
Where:
- method: is the search method to be used. It can be one of the following:
    - B: Breadth-first search
    - I: Iterative deepening search
    - U: Uniform-cost search
    - A: A* search
    - G: Greedy best-first search
    - H: Hill Climbing
- board: is the initial board configuration. It must be 9 digits separated by a blank space from 0 to 8, where 0 represents the empty tile.
- PRINT: is an optional parameter that can be used to print the solution path. It is represented by "PRINT"

Example:
```
    python3 tp1.py H 1 2 3 4 0 5 7 8 6 PRINT
    
    The above represents the following board:
    
    1 2 3
    4 0 5
    7 8 6

    which will be solved using the Hill Climbing algorithm and the solution path will be printed.

