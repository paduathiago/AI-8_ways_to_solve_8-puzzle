import numpy as np
import sys

def read_input():
    alg = sys.argv[1]
    matrix = np.array(sys.argv[2:11]).reshape(3,3).astype(int)
    is_there_print = True if len(sys.argv) == 12 else False
    return alg, matrix, is_there_print

read_input_A = read_input()
for i in range(0,3):
    print(read_input_A[i])

