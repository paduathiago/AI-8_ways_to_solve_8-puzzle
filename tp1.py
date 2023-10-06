import sys
import numpy as np
import queue

class Node:
    def __init__(self, matrix, parent, depth, cost):
        self.state = matrix
        self.parent = parent
        self.depth = depth
        self.cost = cost

    def generate_childen_nodes(self):
        children_nodes = []
        for i in range(len(self.state)):
            for j in range(len(self.state[i])):
                if self.state[i][j] == 0:
                    if i > 0:
                        new_node = Node(np.copy(self.state), self, self.depth + 1, self.cost + 1)
                        children_nodes.append(new_node)
                        children_nodes[-1].state[i][j] = children_nodes[-1].state[i-1][j]
                        children_nodes[-1].state[i-1][j] = 0
                    if i < len(self.state) - 1:
                        new_node = Node(np.copy(self.state), self, self.depth + 1, self.cost + 1)
                        children_nodes.append(new_node)
                        children_nodes[-1].state[i][j] = children_nodes[-1].state[i+1][j]
                        children_nodes[-1].state[i+1][j] = 0
                    if j > 0:
                        new_node = Node(np.copy(self.state), self, self.depth + 1, self.cost + 1)
                        children_nodes.append(new_node)
                        children_nodes[-1].state[i][j] = children_nodes[-1].state[i][j-1]
                        children_nodes[-1].state[i][j-1] = 0
                    if j < len(self.state[i]) - 1:
                        new_node = Node(np.copy(self.state), self, self.depth + 1, self.cost + 1)
                        children_nodes.append(new_node)
                        children_nodes[-1].state[i][j] = children_nodes[-1].state[i][j+1]
                        children_nodes[-1].state[i][j+1] = 0
                    return children_nodes

def read_input():
    alg = sys.argv[1]
    puzzle = np.array(sys.argv[2:11]).reshape(3,3).astype(int)
    is_there_print = sys.argv[11] if len(sys.argv) == 12 else False
    if is_there_print != 'PRINT':
        print("invalid command line arguments")
        return None
    else:
        return alg, puzzle, is_there_print

def print_state(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                print(' ', end=' ')
            else:
                print(matrix[i][j], end=' ')
        print()

def is_goal(node):
    goal = np.array([[1,2,3],[4,5,6],[7,8,0]])
    return np.array_equal(node.state, goal)

def bfs(root_node):
    if is_goal(root_node):
        return root_node
    
    frontier = queue.Queue()
    explored = set()
    frontier.put(root_node)

    while not frontier.empty():
        current_node = frontier.get()
        explored.add(current_node)
        children_nodes = current_node.generate_childen_nodes()
        for child in children_nodes:
            if child not in explored and child not in frontier.queue:
                if is_goal(child):
                    return child
                frontier.put(child)
    
    return None

def print_result(root_node, is_there_print=False):
    operations = queue.LifoQueue()
    node = bfs(root_node)
    total_cost = node.cost
    
    if is_there_print:
        while node != None:
            operations.put(node)
            node = node.parent
        while not operations.empty():
            print_state(operations.get().state)
            print()
            
    print(total_cost)

"""
def process_input(alg, matrix, is_there_print):
    if alg == 'A':
        return alg_A(matrix, is_there_print)
    elif alg == 'B':
        bfs(read_input()[1])
    elif alg == 'G':
    elif alg == 'H':
    elif alg == 'I':
    elif alg == 'U':
"""
read_input_A = read_input()
root_node = Node(read_input()[1], None, 0, 0)
"""print_state(read_input_A[1])
print("-----", end='\n\n')
for i in root_node.generate_childen_nodes():
    print_state(i.state)
    print("-----")"""

print_result(root_node, is_there_print=read_input_A[2])