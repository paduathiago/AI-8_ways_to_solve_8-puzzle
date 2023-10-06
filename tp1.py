import sys
import numpy as np
import queue

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

def generate_childen_nodes(matrix, node):
    children_nodes = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                if i > 0:
                    new_node = Node(np.copy(matrix), node, node.depth + 1, node.cost + 1)
                    children_nodes.append(new_node)
                    children_nodes[-1].state[i][j] = children_nodes[-1].state[i-1][j]
                    children_nodes[-1].state[i-1][j] = 0
                if i < len(matrix) - 1:
                    new_node = Node(np.copy(matrix), node, node.depth + 1, node.cost + 1)
                    children_nodes.append(new_node)
                    children_nodes[-1].state[i][j] = children_nodes[-1].state[i+1][j]
                    children_nodes[-1].state[i+1][j] = 0
                if j > 0:
                    new_node = Node(np.copy(matrix), node, node.depth + 1, node.cost + 1)
                    children_nodes.append(new_node)
                    children_nodes[-1].state[i][j] = children_nodes[-1].state[i][j-1]
                    children_nodes[-1].state[i][j-1] = 0
                if j < len(matrix[i]) - 1:
                    new_node = Node(np.copy(matrix), node, node.depth + 1, node.cost + 1)
                    children_nodes.append(new_node)
                    children_nodes[-1].state[i][j] = children_nodes[-1].state[i][j+1]
                    children_nodes[-1].state[i][j+1] = 0
                return children_nodes

def is_goal(node):
    goal = np.array([[1,2,3],[4,5,6],[7,8,0]])
    return np.array_equal(node.state, goal)

class Node:
    def _init_(self, matrix, parent, depth, cost):
        self.state = matrix
        self.parent = parent
        self.depth = depth
        self.cost = cost

def bfs(matrix):
    node = Node(matrix, None, 0, 0)
    if is_goal(node):
        return node
    
    frontier = queue.Queue()
    explored = set()
    frontier.put(node)

    while not frontier.empty():
        current_node = frontier.get()
        explored.add(current_node)
        children_nodes = generate_childen_nodes(current_node.state, current_node)
        for child in children_nodes:
            if child not in explored and child not in frontier.queue:
                if is_goal(child):
                    return child
                frontier.put(child)
    
    return None

def print_result():
    node = bfs(read_input()[1])
    print(node.cost)

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
print("-----", end='\n\n')
for i in generate_childen_nodes(read_input_A[1], Node(read_input_A[1], None, 0, 0)):
    print_state(i.state)
    print("-----")

print_result()