#####################################################
# CS 6601 - Assignment 1͏︍͏︆͏󠄁
# bfs.py͏︍͏︆͏󠄁
#####################################################

# DO NOT ADD OR REMOVE ANY IMPORTS FROM THIS FILE͏︍͏︆͏󠄁
import math

# Credits if any͏︍͏︆͏󠄁
# 1)͏︍͏︆͏󠄁
# 2)͏︍͏︆͏󠄁
# 3)͏︍͏︆͏󠄁

def return_your_name() -> str:
    """Return your first and last name from this function as a string"""

    return "Justin Kupa"


def breadth_first_search(graph, start, goal) -> list:
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start: Key for the start node.
        goal: Key for the end node.

    Returns:
        The best path via BFS as a list from the start to the goal node (including both).
    """

    # TODO: finish this function!͏︍͏︆͏󠄁
    if start == goal:
        return []
    
    q = []
    visited = set()
    q.append([start])
    visited.add(start)

    while q:
        cur_path = q.pop()
        cur_node = cur_path[-1]

        tmp_path = cur_path
        #sorting in alpha order as instructions state - assuming return value is a list
        sorted_neighbors = sorted(graph.neighbors(cur_node))
        for nei_node in sorted_neighbors:
            if nei_node in visited: continue
            visited.add(nei_node)
            tmp_path.append(nei_node)
            if nei_node == goal:
                return tmp_path
            q.insert(0, tmp_path.copy())
            tmp_path.pop()

    return -1