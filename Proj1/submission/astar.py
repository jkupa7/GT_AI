#####################################################
# CS 6601 - Assignment 1͏︍͏︆͏󠄁
# astar.py͏︍͏︆͏󠄁
#####################################################

# DO NOT ADD OR REMOVE ANY IMPORTS FROM THIS FILE͏︍͏︆͏󠄁
import math
from submission.priority_queue import PriorityQueue

# Credits if any͏︍͏︆͏󠄁
# 1)͏︍͏︆͏󠄁
# 2)͏︍͏︆͏󠄁
# 3)͏︍͏︆͏󠄁

def null_heuristic(graph, u, v):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        u: Key for the first node to calculate from.
        v: Key for the second node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, u, v):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        u: Key for the first node to calculate from.
        v: Key for the second node to calculate to.

    Returns:
        Euclidean distance between the u node and the v node
        Round the result to 3 decimal places (if applicable)
    """

    # TODO: finish this function!͏︍͏︆͏󠄁
    x1, y1 = graph.nodes[u]['pos']
    x2, y2 = graph.nodes[v]['pos']

    x = (x2 - x1) ** 2
    y = (y2 - y1) ** 2

    euc_dist = math.sqrt(x+y)

    return round(euc_dist, 3)

    raise NotImplementedError


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic) -> list:
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start: Key for the start node.
        goal: Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path via A* as a list from the start to the goal node (including both).
    """

    # TODO: finish this function!͏︍͏︆͏󠄁

    if start == goal:
        return []
    
    explored = set()
    best = {}
    q = PriorityQueue()
    h = euclidean_dist_heuristic(graph, start, goal)
    q.append((h, (start, [start], 0)))
    best[start] = 0

    while q.size() != 0:

        cur_heu, cur_tup = q.pop()
        cur_node, cur_path, cur_g = cur_tup

        if cur_node == goal:
            return cur_path
        
        if cur_node in explored: continue
        
        
        sorted_neighbors = sorted(graph.neighbors(cur_node))
        explored.add(cur_node)
        
        tmp_path = cur_path

        for nei in sorted_neighbors:
            best_wei = cur_g + graph.get_edge_weight(cur_node, nei)
            if nei not in best or best_wei < best[nei]:
                best[nei] = best_wei
                nei_heu = best_wei + euclidean_dist_heuristic(graph, nei, goal)
                tmp_path.append(nei)
                q.append((nei_heu, (nei, tmp_path.copy(), best_wei)))
                tmp_path.pop()

    return -1
