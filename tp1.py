import numpy as np
import sys

def read_input():
    alg = sys.argv[1]
    matrix = np.array(sys.argv[2:11]).reshape(3,3).astype(int)
    is_there_print = sys.argv[11] if len(sys.argv) == 12 else False
    return alg, matrix, is_there_print

def print_state(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                print(' ', end=' ')
            else:
                print(matrix[i][j], end=' ')
        print()

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

