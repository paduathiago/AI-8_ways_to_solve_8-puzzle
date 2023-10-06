import numpy as np
import sys

def read_input():
    alg = sys.argv[1]
    puzzle = np.array(sys.argv[2:11]).reshape(3,3).astype(int)
    is_there_print = sys.argv[11] if len(sys.argv) == 12 else False
    return alg, puzzle, is_there_print

def print_state(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                print(' ', end=' ')
            else:
                print(matrix[i][j], end=' ')
        print()

def generate_childen_nodes(matrix):
    children_nodes = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                if i > 0:
                    children_nodes.append(np.copy(matrix))
                    children_nodes[-1][i][j] = children_nodes[-1][i-1][j]
                    children_nodes[-1][i-1][j] = 0
                if i < len(matrix) - 1:
                    children_nodes.append(np.copy(matrix))
                    children_nodes[-1][i][j] = children_nodes[-1][i+1][j]
                    children_nodes[-1][i+1][j] = 0
                if j > 0:
                    children_nodes.append(np.copy(matrix))
                    children_nodes[-1][i][j] = children_nodes[-1][i][j-1]
                    children_nodes[-1][i][j-1] = 0
                if j < len(matrix[i]) - 1:
                    children_nodes.append(np.copy(matrix))
                    children_nodes[-1][i][j] = children_nodes[-1][i][j+1]
                    children_nodes[-1][i][j+1] = 0
                return children_nodes

"""
def process_input(alg, matrix, is_there_print):
    if alg == 'A':
        return alg_A(matrix, is_there_print)
    elif alg == 'B':
    
    elif alg == 'G':
    elif alg == 'H':
    elif alg == 'I':
    elif alg == 'U':
"""

read_input_A = read_input()
print_state(read_input_A[1])
print(generate_childen_nodes(read_input_A[1]))

