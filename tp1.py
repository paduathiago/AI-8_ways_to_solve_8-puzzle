import sys
import numpy as np
import queue
import heapq


class Node:
    def __init__(self, matrix, parent, depth, cost):
        self.state = matrix
        self.parent = parent
        self.depth = depth
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

    def generate_childen_nodes(self):
        children_nodes = []
        for i in range(len(self.state)):
            for j in range(len(self.state[i])):
                if self.state[i][j] == 0:
                    if i > 0:
                        new_node = Node(np.copy(self.state),
                                        self, self.depth + 1, self.cost + 1)
                        children_nodes.append(new_node)
                        children_nodes[-1].state[i][j] = children_nodes[-1].state[i-1][j]
                        children_nodes[-1].state[i-1][j] = 0
                    if i < len(self.state) - 1:
                        new_node = Node(np.copy(self.state),
                                        self, self.depth + 1, self.cost + 1)
                        children_nodes.append(new_node)
                        children_nodes[-1].state[i][j] = children_nodes[-1].state[i+1][j]
                        children_nodes[-1].state[i+1][j] = 0
                    if j > 0:
                        new_node = Node(np.copy(self.state),
                                        self, self.depth + 1, self.cost + 1)
                        children_nodes.append(new_node)
                        children_nodes[-1].state[i][j] = children_nodes[-1].state[i][j-1]
                        children_nodes[-1].state[i][j-1] = 0
                    if j < len(self.state[i]) - 1:
                        new_node = Node(np.copy(self.state),
                                        self, self.depth + 1, self.cost + 1)
                        children_nodes.append(new_node)
                        children_nodes[-1].state[i][j] = children_nodes[-1].state[i][j+1]
                        children_nodes[-1].state[i][j+1] = 0
                    return children_nodes

class InformedNode(Node):
    def __init__(self, matrix, parent, depth, cost, heuristic):
        super().__init__(matrix, parent, depth, cost)
        self.heuristic = heuristic
    

    def __lt__(self, other):
        return super().__lt__(other)


    def generate_childen_nodes(self):
        children_nodes = super().generate_childen_nodes()
        
        for child in children_nodes:
            child.__class__ = InformedNode
            child.heuristic = self.heuristic
            child.cost = self.heuristic(child)
        
        return children_nodes
        

def read_input():
    alg = sys.argv[1]
    puzzle = np.array(sys.argv[2:11]).reshape(3, 3).astype(int)
    is_there_print = sys.argv[11] if len(sys.argv) == 12 else False
    if is_there_print != 'PRINT':
        print("invalid command line arguments")
        return None
    else:
        return alg, puzzle, is_there_print


def process_input(alg, matrix, is_there_print):  # Refactor: print result in main
    root_node = Node(matrix, None, 0, 0)
    if alg == 'A':
        root_node = InformedNode(matrix, None, 0, 0, manhattan_distance_heuristic)
        result = a_star_search(root_node)
    elif alg == 'B':
        result = bfs(root_node)
    elif alg == 'G':
        root_node = InformedNode(matrix, None, 0, 0, misplaced_tiles_heuristic)
        result = greedy_best_first_search(root_node)
    elif alg == 'H':
        pass
    elif alg == 'I':
        result = iterative_deepening_search(root_node)
    elif alg == 'U':
        result = uniform_cost_search(root_node)  
    else:
        print("invalid algorithm")
        return
    
    print_result(result, is_there_print)


def print_state(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                print(' ', end=' ')
            else:
                print(matrix[i][j], end=' ')
        print()


def is_goal(node):  # Check if there are any other goal states
    goal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    return np.array_equal(node.state, goal)


def bfs(root_node):
    if is_goal(root_node):
        return root_node

    frontier = queue.Queue()
    explored = []
    frontier.put(root_node)

    while not frontier.empty():
        current_node = frontier.get()
        explored.append(current_node.state)
        children_nodes = current_node.generate_childen_nodes()
        for child in children_nodes:
            if not any(np.all(np.equal(child.state, element)) for element in explored) and \
            not any(np.all(np.equal(child.state, node.state)) for node in frontier.queue):
                if is_goal(child):
                    return child
                frontier.put(child)

    return None


def uniform_cost_search(root_node):
    frontier = [root_node]
    heapq.heapify(frontier)
    explored = []

    while frontier:  # while frontier is not empty
        current_node = heapq.heappop(frontier)
        if is_goal(current_node):
            return current_node
        
        explored.append(current_node.state)
        children_nodes = current_node.generate_childen_nodes()
        
        for child in children_nodes:
            if not any(np.all(np.equal(child.state, element)) for element in explored) and \
            not any(np.all(np.equal(child.state, node.state)) for node in frontier):
                heapq.heappush(frontier, child)
            elif any(np.all(np.equal(child.state, node.state)) for node in frontier):
                for node in frontier:
                    if np.all(np.equal(child.state, node.state)):
                        if child.cost < node.cost:
                            frontier.remove(node)
                            heapq.heappush(frontier, child)

    return None


def is_cycle(node):
    current_node = node.parent
    
    while current_node != None:
        if np.array_equal(node.state, current_node.state):
            return True
        current_node = current_node.parent
    
    return False


def iterative_deepening_search(root_node):
    depth = 0
    while True:
        result = depth_limited_search(root_node, depth)
        if result != 'cutoff':
            return result
        depth += 1


def depth_limited_search(root_node, limit):
    frontier = queue.LifoQueue()
    frontier.put(root_node)
    result = 'failure'

    while not frontier.empty():
        current_node = frontier.get()
        
        if is_goal(current_node):
            return current_node
        
        if current_node.depth > limit:
            result = 'cutoff'
        elif not is_cycle(current_node):
            for child in current_node.generate_childen_nodes():
                frontier.put(child)
    
    return result
            

def misplaced_tiles_heuristic(node):
    goal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    return np.count_nonzero(node.state != goal)


def manhattan_distance_heuristic(node):
    goal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    distance = 0
    
    for i in range(len(node.state)):
        for j in range(len(node.state[i])):
            if node.state[i][j] != 0:
                distance += np.sum(np.abs(np.subtract(np.where(node.state == node.state[i][j]), np.where(goal == node.state[i][j]))))
    
    # f(n) = g(n) + h(n), where g(n) is the depth(real cost to get to the current node) and h(n) is the heuristic
    return node.depth + distance  


def greedy_best_first_search(root_node):
    return uniform_cost_search(root_node)
        

def a_star_search(root_node):
    return uniform_cost_search(root_node)


def print_result(result_node, is_there_print=False):
    operations = queue.LifoQueue()
    total_cost = result_node.depth
    current_node = result_node

    if is_there_print:
        while current_node != None:
            operations.put(current_node)
            current_node = current_node.parent
        while not operations.empty():
            print_state(operations.get().state)
            print()

    print(total_cost, end='\n')


def main():
    process_input(*read_input())


main()
