import os, sys
import networkx as nx
from mazelib import Maze
from mazelib.generate.Prims import Prims
import numpy as np
import random as rand

import matplotlib.pyplot as plt
from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")

import plotter as pter # Adds AIMA toolbox to path
# AIMA_TOOLBOX_ROOT="/home/valeria/Documents/Code/aima-python-uofg_v20202021a"
# sys.path.append(AIMA_TOOLBOX_ROOT)

import search as sch
# from notebookutils import show_map#, display_visual

#%% HELPER FUNCTIONS

def square2flat(i, j, N):
    """From indices in a square array, to index in flatten array"""
    return i*N + j

def flat2square(k, N):
    """From indices in a flatten array, to index in square array"""
    return (k // N, k % N)

def square_neighbours(k, N):
    """Return neighbours in a squared array while working with flatten indices"""
    i, j = flat2square(k, N)
    neighbours = []
    if i>0: neighbours.append(k-N)
    if i<N-1: neighbours.append(k+N)
    if j>0: neighbours.append(k-1)
    if j<N-1: neighbours.append(k+1)
    return neighbours

tile2str = lambda i, j : f"S_{i:02d}_{j:02d}"

#%% MAZE GENERATION

def wormholes_maze_structure(N, M, show_plots=False):
    """Creates a NxN maze with M wormholes or teleportation links"""

    assert N % 2 == 1, "N needs to be an odd number"

    # Generate a random maze with mazelib
    m = Maze( )
    m.generator = Prims(int(N/2), int(N/2))
    m.generate()
    m.generate_entrances(True, True)

    # Extract the maze grid and add the entrances
    maze_grid = np.logical_not(m.grid)*2
    maze_grid[m.start] = 1
    maze_grid[m.end] = 3
    maze_grid = maze_grid.astype(np.uint8)

    if M>0:

        # Determine available tiles for wormhole entrances
        indices_available = np.where(maze_grid.flatten()==2)[0].tolist()
        n_available = len(indices_available)
        assert n_available >= 2*M, "Not enough empty tiles for that many wormholes"

        # Randomize positions
        rand.shuffle(indices_available)

        # Pick wormhole entrance and exit, making sure they are not neighbours
        wormholes = []
        filled = False
        while not filled:
            if len(wormholes)%2 == 0:
                add_candidate = True
            else:
                if indices_available[0] not in square_neighbours(wormholes[-1], N):
                    add_candidate = True
                else: 
                    add_candidate = False
            if add_candidate:
                wormholes.append( indices_available.pop(0) )
            else:
                indices_available.append( indices_available.pop(0) )
            filled = len(wormholes) == 2*M
        
    else: wormholes = []

    # Optional: show map
    if show_plots:
        fontsize = "large" if N<=13 else "medium"
        plt.imshow(maze_grid, cmap="gray", origin="lower")
        for w_k in range(M):
            i_start, j_start = flat2square(wormholes[2*w_k], N)
            i_end, j_end = flat2square(wormholes[2*w_k+1], N)
            plt.arrow(j_start, i_start, j_end-j_start, i_end-i_start,
                    width=0.05, head_width=0.3, length_includes_head=True,
                    color="C3")
        for (j, i), label, color in zip([m.start, m.end], ["S","E"], ["w", "k"]):
            plt.text(i-.1, j-.1, label, color=color, fontsize=fontsize)

    return maze_grid, wormholes

class GraphProblem(sch.GraphProblem):

    """The problem of searching a graph from one node to another.
    
    Credits to the 'Artificial Intelligence: a Modern Approach' authors 
    on whose repository this class is based.

    Sole difference with the original is that the heuristic function h 
    accepts optional arguments"""

    def h(self, node, *args):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(sch.distance(locs[node], locs[self.goal]))

            return int(sch.distance(locs[node.state], locs[self.goal]))
        else:
            return sch.infinity

class WormholesGraphProblem(GraphProblem):
    """The problem of searching a graph with teleportation links.
    
    An agent at the start of a teleportation link can choose to go down the wormhole.
    However, it cannot know in advance where the exit is
    
    The included heuristic function $h$ is consistent if the graph has uniform cost $c$.
    - For any tile $n_i$ that is neither the entrance nor the exit of a wormhole, 
    $h(n_i)$ returns the Euclidean distance to the goal
    - For any tile $n_i$ that is the entrance of a wormhole, the heuristic function 
    returns $h(n_i)=h(n_{i-1})-c$, so that $f(n_i)=f(n_{i-1})$ matches its parent's 
    because $f(n_i)=h(n_i)+g(n_i)=h(n_i)+g(n_{i-1})+c$
    - For any tile $n_i$ that is the exit of a wormhole, the heuristic function 
    returns $h(n_i)=h(n_{i-2})-2c$, so that $f(n_i)=f(n_{i-2})$ matches its 
    grandparent's because $f(n_i)=h(n_i)+g(n_i)=h(n_i)+g(n_{i-1})+c=h(n_i)+g(n_{i-2})+2c$
    """

    def __init__(self, start, end, graph, wormholes, verbose=False):
        self.step_cost = list(list(graph.graph_dict.values())[0].values())[0]
        self.check_uniform_cost(graph)
        super().__init__(start, end, graph)
        self.wormholes = wormholes
        self.verbose = verbose

    def check_uniform_cost(self, graph):
        correct = True
        for connections in graph.graph_dict.values():
            for step_cost in connections.values():
                correct = correct and step_cost == self.step_cost
        if not correct:
            raise ValueError("Proposed map does not have uniform cost")

    def is_wormhole(self, node, parent):
        return self.is_wormhole_entrance(node) or self.is_wormhole_link(node, parent)

    def is_wormhole_entrance(self, node):
        if type(node) is not str:
            node = node.state
        if node in self.wormholes[::2]:
            if self.verbose: print("Teleportation link entrance")
            return True
        return False

    def is_wormhole_link(self, node, parent):
        if type(node) is not str: node = node.state
        if parent is not None and node in self.wormholes:
            index = self.wormholes.index(node)
            if type(parent) is not str: parent = parent.state
            if index>0 and parent==self.wormholes[index-1]:
                if self.verbose: print("Teleportation link")
                return True
        return False

    def distance_to_goal(self, node):
        if type(node) is not str: node = node.state
        position_node = np.array(self.graph.locations[node])
        position_goal = np.array(self.graph.locations[self.goal])
        distance = float( np.sqrt( np.sum( (position_goal - position_node)**2 ) ) )
        if self.verbose: 
            print("Node position", position_node)
            print("Distance to the goal", distance)
        return distance

    def h(self, node, parent):
        """Euclidean distance, except for wormholes"""
        if type(node) is not str: node = node.state
        if not self.is_wormhole(node, parent):
            return self.distance_to_goal(node)
        elif self.is_wormhole_entrance(node):
            parent_h = self.distance_to_goal(parent)
            print("Parent had h", parent_h, "so h is set to be", parent_h - self.step_cost)
            return parent_h - self.step_cost
        else:
            grandparent_h = self.distance_to_goal(parent.parent)
            print("Grandprarent had h", grandparent_h, "so h is set to be", 
                  grandparent_h - 2*self.step_cost)
            return grandparent_h - 2*self.step_cost

def wormholes_maze_grid(N, M, show_plots=False):

    maze_grid, wormholes = wormholes_maze_structure(N, M, show_plots)

    maze_dict = {}
    map_locations = {}
    for i in range(N):
        for j in range(N):
            if maze_grid[i,j]: 
                cell_dict = {}
                if i>0 and maze_grid[i-1,j]: cell_dict[ tile2str(j,i-1) ] = 1
                if i<N-1 and maze_grid[i+1,j]: cell_dict[ tile2str(j,i+1) ] = 1
                if j>0 and maze_grid[i,j-1]: cell_dict[ tile2str(j-1,i) ] = 1
                if j<N-1 and maze_grid[i,j+1]: cell_dict[ tile2str(j+1,i) ] = 1
                maze_dict[ tile2str(j,i) ] = cell_dict
                map_locations[ tile2str(j,i) ] = (j,i)

    maze_start = tuple([ int(i) for i in np.where(maze_grid==1) ])
    maze_end = tuple([ int(i) for i in np.where(maze_grid==3) ])
    maze_start = tile2str(*maze_start[::-1])
    maze_end = tile2str(*maze_end[::-1])

    for w_k in range(M):
        i_start, j_start = flat2square(wormholes[2*w_k], N)
        i_end, j_end = flat2square(wormholes[2*w_k+1], N)
        maze_dict[ tile2str(j_start,i_start) ][ tile2str(j_end,i_end) ] = 1
        
    teleportation_links = []
    for w_ij in wormholes:
        i, j = flat2square(w_ij, N)
        teleportation_links.append( tile2str(j,i) )
    
    maze_map = sch.Graph(maze_dict)
    maze_map.locations = map_locations
    
    return maze_start, maze_end, maze_map, teleportation_links

def wormholes_maze_problem(N, M, show_plots=False):

    maze_data = wormholes_maze_grid(N, M, show_plots)
    maze_problem = WormholesGraphProblem(*maze_data, verbose=True)

    return maze_problem

def get_wormholes_maze_graphic_data(maze_problem, node_colors=None):

    if node_colors is None:
        node_colors = {node: 'white' for node in maze_problem.graph.locations.keys()}
    node_positions = maze_problem.graph.locations
    node_label_pos = { k:[v[0],v[1]-.3]  for k,v in maze_problem.graph.locations.items() }
    edge_weights = {(k, k2) : v2 for k, v in maze_problem.graph.graph_dict.items() for k2, v2 in v.items()}

    maze_graph_data = { 'graph_dict' : maze_problem.graph.graph_dict,
                        'node_colors': node_colors,
                        'node_positions': node_positions,
                        'node_label_positions': node_label_pos,
                        'edge_weights': edge_weights
                       }

    return maze_graph_data

def show_wormholes_maze_problem(maze_problem, node_colors=None, iterations=None):

    maze_graph_data = get_wormholes_maze_graphic_data(maze_problem, node_colors)
    
    title = f"{iterations:.0f} iterations" if iterations is not None else None
    return pter.show_map(maze_graph_data, title=title)

#%% OLD VERSIONS

class WormholesGraphProblemV1(sch.GraphProblem):
    """The problem of searching a graph with teleportation links.
    
    An agent at the start of a teleportation link can choose to go down the wormhole.
    However, it cannot know in advance where the exit is"""

    def __init__(self, start, end, map, wormholes, verbose=False):
        super().__init__(start, end, map)
        self.wormholes = wormholes
        self.verbose = verbose

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def is_wormhole(self, node, parent):
        if type(node) is not str:
            node = node.state
        if parent and type(parent) is not str:
            parent = parent.state
        if parent and node in self.wormholes:
            index = self.wormholes.index(node)
            if index>0 and parent==self.wormholes[index-1]:
                if self.verbose: print("Teleportation link")
                return True
        return False

    def h(self, node, is_unknown=False):
        """Euclidean distance, except for wormholes"""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is not str:
                node = node.state
            position_node = np.array(locs[node])
            position_goal = np.array(locs[self.goal])
            if self.verbose: print("Node position", position_node)
            # The most optimistic assumption is that a wormhole will take you right adjacent to the goal
            if is_unknown:
                if self.verbose: print("Unknown distance to the goal")
                return 1 
            if self.verbose: print("Distance to the goal", float( np.sqrt( np.sum( (position_goal - position_node)**2 ) ) ))
            return float( np.sqrt( np.sum( (position_goal - position_node)**2 ) ) )
        else:
            return sch.infinity

class WormholesGraphProblemV0(sch.GraphProblem):
    """The problem of searching a graph with teleportation links.
    
    An agent at the start of a teleportation link can choose to go down the wormhole.
    However, it cannot know in advance where the exit is.
    """

    def __init__(self, start, end, map, wormholes, verbose=False):
        super().__init__(start, end, map)
        self.wormholes = wormholes
        self.verbose = verbose

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def h(self, node, parent=None):
        """Euclidean distance, except for wormholes"""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is not str:
                node = node.state
            if parent and type(parent) is not str:
                parent = parent.state
            position_node = np.array(locs[node])
            position_goal = np.array(locs[self.goal])
            if self.verbose: print("Node position", position_node)
            # The most optimistic assumption is that a wormhole will take you right adjacent to the goal
            if parent and node in self.wormholes:
                index = self.wormholes.index(node)
                if index>0 and parent==self.wormholes[index-1]:
                    if self.verbose: print("Teleportation link")
                    return 1 
            if self.verbose: print("Distance to the goal", float( np.sqrt( np.sum( (position_goal - position_node)**2 ) ) ))
            return float( np.sqrt( np.sum( (position_goal - position_node)**2 ) ) )
            #     print("Distance to the goal", int( np.sum( np.abs( position_goal - position_node) ) ))
            # return int( np.sum( np.abs( position_goal - position_node) ) )
            # The sum of the horizontal and vertical distance to the goal, except for wormholes
        else:
            return sch.infinity
    
def wormholes_maze_problem_v0(N, M, show_plots=False, verbose=True):

    maze_data = wormholes_maze_grid(N, M, show_plots)
    maze_problem = WormholesGraphProblemV0(*maze_data, verbose=verbose)

    return maze_problem