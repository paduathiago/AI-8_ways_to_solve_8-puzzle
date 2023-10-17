import sys
import numpy as np
import queue
import heapq
import copy


class Node:
    def __init__(self, matrix, parent, depth, cost):
        self.state = matrix
        self.parent = parent
        self.depth = depth
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        if other.__class__ != self.__class__:
            return False
        return np.array_equal(self.state, other.state)

    def generate_children_nodes(self):
        children_nodes = []
        i, j = np.where(self.state == 0)
        i, j = i[0], j[0]

        moves = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
        
        for new_i, new_j in moves:
            if 0 <= new_i < len(self.state) and 0 <= new_j < len(self.state[0]):
                new_state = np.copy(self.state)
                new_state[i][j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[i][j]
                new_node = Node(new_state, self, self.depth + 1, self.cost + 1)
                children_nodes.append(new_node)

        return children_nodes


class InformedNode(Node):
    def __init__(self, matrix, parent, depth, cost, heuristic):
        super().__init__(matrix, parent, depth, cost)
        self.heuristic = heuristic
        self.cost = self.heuristic(self)


    def generate_children_nodes(self):
        children_nodes = super().generate_children_nodes()
        
        for child in children_nodes:
            child.__class__ = InformedNode
            child.heuristic = self.heuristic
            child.cost = child.heuristic(child)
        
        return children_nodes
        

def read_input():
    alg = sys.argv[1]
    puzzle = np.array(sys.argv[2:11]).reshape(3, 3).astype(int)
    is_there_print = True if len(sys.argv) == 12 else False
    if len(sys.argv) == 12 and sys.argv[11] != 'PRINT':
        print("invalid command line arguments")
        return None
    else:
        return alg, puzzle, is_there_print


def process_input(alg, matrix):
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
        root_node = InformedNode(matrix, None, 0, 0, manhattan_distance_heuristic)
        result = hill_climbing_search(root_node)
    elif alg == 'I':
        result = iterative_deepening_search(root_node)
    elif alg == 'U':
        result = uniform_cost_search(root_node)  
    else:
        print("invalid algorithm")
        return None
    
    return result


def print_state(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                print(' ', end=' ')
            else:
                print(matrix[i][j], end=' ')
        print()


def is_goal(node): 
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
        explored.append(current_node)
        children_nodes = current_node.generate_children_nodes()
        for child in children_nodes:
            if not child in explored and not child in frontier.queue:
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
        
        explored.append(current_node)
        children_nodes = current_node.generate_children_nodes()
        
        for child in children_nodes:           
            try:
                child_index = frontier.index(child)
            except ValueError:
                child_index = -1
            
            if not child in explored and child_index == -1:
                heapq.heappush(frontier, child)
            elif child_index != -1:
                if child.cost < frontier[child_index].cost:
                    frontier.pop(child_index)
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
            for child in current_node.generate_children_nodes():
                frontier.put(child)
    
    return result
            

def misplaced_tiles_heuristic(node):
    goal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    mask = node.state != 0
    return np.count_nonzero(mask & (node.state != goal)) 


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


def best_neighbor(node):
    best_neighbors = []
    neighbors = node.generate_children_nodes()
    neighbors.sort(key=lambda x: x.cost)
    for i in range(len(neighbors)):
        if i > 0:
            if neighbors[i].cost > neighbors[i-1].cost:
                break
        best_neighbors.append(neighbors[i])

    return best_neighbors


def hill_climbing_search(root_node):
    current_node = root_node
    K = 150
    while K:
        neighbor = best_neighbor(current_node)
        neighbor.cost -= neighbor.depth

        if neighbor.cost > current_node.cost:
            return current_node
        elif neighbor.cost < current_node.cost:
            K = 150
        elif neighbor.cost == current_node.cost:
            K -= 1
        current_node = copy.deepcopy(neighbor)
        
    
    return current_node


def print_result(result_node, is_there_print=False):
    operations = queue.LifoQueue()
    total_cost = result_node.depth
    current_node = result_node

    print(total_cost, end='\n\n')

    if is_there_print:
        while current_node != None:
            operations.put(current_node)
            current_node = current_node.parent
        while not operations.empty():
            print_state(operations.get().state)
            print()


def main():
    alg, puzzle, is_there_print = read_input()
    result = process_input(alg, puzzle)
    print_result(result, is_there_print)


main()
