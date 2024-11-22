# import os, sys
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import maze as m # Adds AIMA toolbox to path
# AIMA_TOOLBOX_ROOT="/home/valeria/Documents/Code/aima-python-uofg_v20202021a"
# sys.path.append(AIMA_TOOLBOX_ROOT)

import heapq
import search as sch
# from notebookutils import show_map, display_visual
from aimautils import PriorityQueue

class DoublePriorityQueue(PriorityQueue):
    """A Queue in which elements are returned ordered according to an f value.
    
    If order is 'min', the item with minimum f is returned first; if order is 
    'max', then it is the item with maximum f. Also supports dict-like lookup.

    In contrast with AIMA's implementation...
    ...This queue doesn't automatically calculate `f(input)`: it needs 
    `f(input), input` as the input to the `append` method
    ...This queue is built to have `input = node, parent`, storing two 
    elements associated to `f(input)` instead of one; i.e. each element in the 
    queue is `f(node), node, parent` instead of just `f(node), node`"""

    def __init__(self, order='min'):
        assert order in ["min", "max"], "Needs to be 'min' or 'max'"
        self.order = order
        self.heap = []

    def append(self, f_value, node, parent):
        """Insert item at its correct position."""
        if self.order == 'min':
            heapq.heappush(self.heap, (f_value, node, parent))
        elif self.order == 'max':  # now item with max f(x)
            heapq.heappush(self.heap, (-f_value, node, parent))  # will be popped first

    def extend(self, f_values, nodes, parents):
        """Insert each item in items at its correct position."""
        for f_value, node, parent in zip(f_values, nodes, parents):
            self.append(f_value, node, parent)

    def pop(self):
        """Pop and return the item with min or max f(x) value
        depending on the order."""
        if self.heap:
            return heapq.heappop(self.heap)
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    def __contains__(self, node):
        """Return True if item in PriorityQueue."""
        nodes = [n for (f, n, p) in self.heap]
        return node in nodes

    def __getitem__(self, node):
        for f, n, p in self.heap:
            if n==node:
                return f, n, p

    def __delitem__(self, node):
        """Delete the first occurrence of key."""
        nodes = [n for (f, n, p) in self.heap]
        index = nodes.index(node)
        self.heap.remove(self.heap[index])
        heapq.heapify(self.heap)

    def update_item(self, f_value, node, parent, new_f_value):
        """Delete the first occurrence of key."""
        index = self.heap.index((f_value, node, parent))
        self.heap.remove(self.heap[index])
        self.heap.append(new_f_value, node, parent)
        heapq.heapify(self.heap)

def wormholes_maze_A_star_solver(maze_problem, verbose=True):
    """Search the nodes with the lowest f scores first.

    The scoring function f(x) is calculating adding up...
    ...the path cost g(x) calculated using the weights in the network links
    ...the heuristic function h(x) returing the euclidean distance for 
    known distances, and 1 for unknown distances such as those in wormholes
    
    Partial credit to the AI Module lecturers: this function has been 
    taken from Lab 3 before being modified.
    
    In contrast with the original...
    ...This function stores both node and parent inside of the queue
    ...It requires `DoublePriorityQueue` to work
    ...It also requires `WormholesGraphProblem`, because it uses its 
    `is_wormhole` and the `is_unknown` optional argument included inside 
    its `h` function
    ...The unknown values are replaced during exploration with known values
    ...This function contains plenty of annotations to describe the progress 
    while running
    """
    
    def comment(*args):
        if verbose: print(*args)
    maze_problem.verbose = verbose

    # we use these two variables at the time of visualisations
    iterations = 0
    comment(">> Iteration", iterations, ">> Setting up the problem")
    all_node_colors = []
    node_colors = {k : 'white' for k in maze_problem.graph.nodes()}
    
    iterations += 1
    comment(">> Iteration", iterations, ">> Adding the start point")
    node = sch.Node(maze_problem.initial)
    node_colors[node.state] = "green"    
    all_node_colors.append(dict(node_colors))
    
    if maze_problem.goal_test(node.state):
        node_colors[node.state] = "limegreen"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        return(iterations, all_node_colors, node)
    
    frontier = DoublePriorityQueue('min')
    frontier.append(0, node, None)
    
    node_colors[node.state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))
    
    explored = set()
    while frontier:

        comment(">> Iteration", iterations, ">> Exploring a node in the frontier")
        f_value, node, parent = frontier.pop()
        comment("== Node", (f_value, node, parent), "popped out")
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        explored.add(node.state)
        if parent and maze_problem.is_wormhole(node, parent):
            comment(">> Iteration", iterations, ">> Updating", node, "with parent", parent)
            h_value = maze_problem.h(node, is_unknown=False)
            updated_f_value = f_value - 1 + h_value # Subtract 1 because that was the h value while still unknown
            comment(frontier.heap)
            frontier.append(updated_f_value, node, parent)
            iterations += 1
        
        if maze_problem.goal_test(node.state):
            comment(">> Iteration", iterations, ">> Goal has been found")
            node_colors[node.state] = "limegreen"
            node_colors[maze_problem.initial] = "limegreen"
            for extra_node in node.solution():
                node_colors[extra_node] = "limegreen"  
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return(iterations, all_node_colors, node)

        comment("Will go over the following childs:", node.expand(maze_problem))
        for child in node.expand(maze_problem):
            comment("== Child", child)
            if child.state not in explored and child not in frontier:
                comment(">> Iteration", iterations, ">> Adding that child to the frontier")
                child_h_value = maze_problem.h(node, is_unknown=maze_problem.is_wormhole(child, node))
                child_f_value = child.path_cost  + child_h_value
                frontier.append(child_f_value, child, node)
                comment("Added", (child_f_value, child, node))
                node_colors[child.state] = "orange"
                iterations += 1
                all_node_colors.append(dict(node_colors))
            elif child in frontier:
                comment(">> Iteration", iterations, ">> Detected that child already inside the frontier")
                incumbent_f_value, incumbent_node, incumbent_parent = frontier[child]
                child_h_value = maze_problem.h(node, is_unknown=maze_problem.is_wormhole(child, node))
                child_f_value = child.path_cost  + child_h_value
                comment("Examining previous", (incumbent_f_value, incumbent_node, incumbent_parent))
                if child_f_value < incumbent_f_value:
                    del frontier[incumbent_node]
                    frontier.append(child_f_value, child, node)
                    comment("= Removed that and added", (child_f_value, child, node), "instead")
                    node_colors[child.state] = "orange"
                    iterations += 1
                    all_node_colors.append(dict(node_colors))
                else:
                    comment("= Kept", (incumbent_f_value, incumbent_node, incumbent_parent))
            else:
                comment("Child had already been explored")

        comment(">> Iteration", iterations, ">> Finished exploring a node")
        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        comment("Current queue", frontier.heap)
    return None

#%% OLD VERSIONS

class MultiplePriorityQueue(PriorityQueue):
    """A Queue in which elements are returned ordered according to f.
    
    If order is 'min', the item with minimum f(x) is returned first; if order is 
    'max', then it is the item with maximum f(x). Also supports dict-like lookup.

    In contrast with AIMA's implementation, this f can take optional arguments"""

    def append(self, item, *args):
        """Insert item at its correct position."""
        heapq.heappush(self.heap, (self.f(item, *args), item))

    def extend(self, items):
        raise NotImplementedError

    def pop(self):
        """Pop and return the item (with min or max f(x) value
        depending on the order."""
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.heap)

    def __contains__(self, item):
        """Return True if item in PriorityQueue."""
        keys = [k for (_, k) in self.heap]
        return item in keys

    def __getitem__(self, key, *args):
        for _, item in self.heap:
            if item == key:
                return item

    def get_full_item(self, key):
        for _, item in self.heap:
            if item == key:
                return (_, item)

    def __delitem__(self, key):
        """Delete the first occurrence of key."""
        keys = [k for (_, k) in self.heap]
        index = keys.index(key)
        self.heap.remove(self.heap[index])
        heapq.heapify(self.heap)

def multiple_priority_A_star_f_solver(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned.
    
    Partial credit to the AI Module lecturers: this function has been 
    taken from Lab 3 before being modified.
    
    In contrast with the original...
    ...This function stores both node and parent inside of the queue
    ...It requires `DoublePriorityQueueV0` to work
    ...It's therefore meant to work on `WormholesGraphProblemV0` that can take 
    both node and parent to calculate the distance.
    """
    
    # we use these two variables at the time of visualisations
    iterations = 0
    all_node_colors = []
    node_colors = {k : 'white' for k in problem.graph.nodes()}
    
    f = sch.memoize(f, 'f')
    node = sch.Node(problem.initial)
    
    node_colors[node.state] = "red"
    iterations += 1
    all_node_colors.append(dict(node_colors))
    
    if problem.goal_test(node.state):
        node_colors[node.state] = "limegreen"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        return(iterations, all_node_colors, node)
    
    # frontier = sch.PriorityQueue('min', f)
    frontier = MultiplePriorityQueue('min', f)
    # frontier.append(node)
    frontier.append(node, None)
    
    node_colors[node.state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))
    
    explored = set()
    while frontier:
        node = frontier.pop()
        
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        
        if problem.goal_test(node.state):
            node_colors[node.state] = "limegreen"
            node_colors[problem.initial] = "limegreen"
            for extra_node in node.solution():
                node_colors[extra_node] = "limegreen"  
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return(iterations, all_node_colors, node)
        
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                # frontier.append(child)
                frontier.append(child, node)
                node_colors[child.state] = "orange"
                iterations += 1
                all_node_colors.append(dict(node_colors))
            elif child in frontier:
                # incumbent = frontier[child]
                current_f, incumbent = frontier.get_full_item(child)
                # if f(child) < f(incumbent):
                if f(child, node) < current_f:
                    del frontier[incumbent]
                    # frontier.append(child)
                    frontier.append(child, node)
                    node_colors[child.state] = "orange"
                    iterations += 1
                    all_node_colors.append(dict(node_colors))

        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))
    return None

def multiple_priority_A_star_h_solver(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass.
    
    Credits to the AI Module lecturers: this function has been 
    taken from Lab 3 and it has not been modified"""
    h = sch.memoize(h or problem.h, 'h')
    iterations, all_node_colors, node = multiple_priority_A_star_f_solver(
        problem, lambda node, parent : node.path_cost + h(node, parent)
    )
    return(iterations, all_node_colors, node)
