#####################################################
# CS 6601 - Assignment 1͏︍͏︆͏󠄁
# ucs.py͏︍͏︆͏󠄁
#####################################################

# DO NOT ADD OR REMOVE ANY IMPORTS FROM THIS FILE͏︍͏︆͏󠄁
import math
from submission.priority_queue import PriorityQueue

# Credits if any͏︍͏︆͏󠄁
# 1)͏︍͏︆͏󠄁
# 2)͏︍͏︆͏󠄁
# 3)͏︍͏︆͏󠄁

def uniform_cost_search(graph, start, goal) -> list:
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start: Key for the start node.
        goal: Key for the end node.

    Returns:
        The best path via UCS as a list from the start to the goal node (including both).
    """

    # TODO: finish this function!͏︍͏︆͏󠄁
    if start == goal:
        return []
    
    explored = set()
    best = {}
    q = PriorityQueue()
    q.append((0, (start, [start])))
    best[start] = 0

    while q.size() != 0:

        cur_wei, cur_tup = q.pop()
        cur_node, cur_path = cur_tup
        if cur_node == goal:
            return cur_path
        if cur_node in explored: continue

        sorted_neighbors = sorted(graph.neighbors(cur_node))
        explored.add(cur_node)
        tmp_path = cur_path
        for nei in sorted_neighbors:
            best_wei = cur_wei + graph.get_edge_weight(cur_node, nei)
            if nei not in best or best_wei < best[nei]: 
                best[nei] = best_wei
                tmp_path.append(nei)
                q.append((best_wei, (nei, tmp_path.copy())))
                tmp_path.pop()
        

    return -1



