#####################################################
# CS 6601 - Assignment 1͏︍͏︆͏󠄁
# bi_ucs.py͏︍͏︆͏󠄁
#####################################################

# DO NOT ADD OR REMOVE ANY IMPORTS FROM THIS FILE͏︍͏︆͏󠄁
import math
from submission.priority_queue import PriorityQueue

# Credits if any͏︍͏︆͏󠄁
# 1)͏︍͏︆͏󠄁
# 2)͏︍͏︆͏󠄁
# 3)͏︍͏︆͏󠄁

def bidirectional_ucs(graph, start, goal) -> list:
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start: Key for the start node.
        goal: Key for the end node.

    Returns:
        The best path via bi-UCS as a list from the start to the goal node (including both).
    """

    if goal == start: return []

    f_q = PriorityQueue()
    r_q = PriorityQueue()
    f, r = start, goal
    node_f = (0, (start, [start]))
    node_r = (0, (goal, [goal]))
    f_q.append(node_f)
    r_q.append(node_r)

    explored_f, explored_r = {start:0}, {goal:0}
    pred_f, pred_r = {start:None}, {goal:None}

    best_path = []
    best_node = (float('inf'), (None, []))

    while (f_q.size() != 0 and r_q.size() != 0):# and not stop_criteria_met(f, explored_f, r, explored_r):

        if f_q.top()[0] + r_q.top()[0] >= best_node[0]: break

        if f_q.top() < r_q.top():
            best_node = proceed('forward', f_q, explored_f, explored_r, pred_f, pred_r, best_node, graph)

            f, best_path = best_node[1][0], best_node[1][1]

        else:
            best_node = proceed('reverse', r_q, explored_f, explored_r, pred_f, pred_r, best_node, graph)

            r, best_path = best_node[1][0], best_node[1][1]

    return best_path
    # TODO: finish this function!͏︍͏︆͏󠄁


    
    raise NotImplementedError

def proceed(dir, q, explored_f, explored_r, pred_f, pred_r, best_node, graph):

    if dir == 'forward':
        f_node = q.pop()
        f_cost, f_tup = f_node
        f, f_path = f_tup
        explored_f[f] = f_cost

        sorted_neighbors = sorted(graph.neighbors(f))

        for nei in sorted_neighbors:
            nei_path_cost = f_cost + graph.get_edge_weight(f, nei)
            if nei not in explored_f or nei_path_cost < explored_f[nei]:
                explored_f[nei] = nei_path_cost
                pred_f[nei] = f
                f_path.append(nei)
                node_nei = (nei_path_cost, (nei, f_path.copy()))
                q.append(node_nei)
                f_path.pop()

                if nei in explored_r:
                    union_node = join_nodes(dir, nei, pred_f, pred_r, explored_f, explored_r)
                    if union_node[0] < best_node[0]:
                        best_node = union_node

    if dir == 'reverse':
        r_node = q.pop()
        r_cost, r_tup = r_node
        r, r_path = r_tup
        explored_r[r] = r_cost

        sorted_neighbors = sorted(graph.neighbors(r))

        for nei in sorted_neighbors:
            nei_path_cost = r_cost + graph.get_edge_weight(r, nei)
            if nei not in explored_r or nei_path_cost < explored_r[nei]:
                explored_r[nei] = nei_path_cost
                pred_r[nei] = r
                r_path.append(nei)
                node_nei = (nei_path_cost, (nei, r_path.copy()))
                q.append(node_nei)
                r_path.pop()

                if nei in explored_f:
                    union_node = join_nodes(dir, nei, pred_f, pred_r, explored_f, explored_r)
                    if  union_node[0] < best_node[0]:
                        best_node = union_node

    return best_node

def join_nodes(dir, crossover, pred_f, pred_r, explored_f, explored_r):

    forward_path, reverse_path = reconstruct_path(crossover, pred_f), reconstruct_path(crossover, pred_r)

    reverse_path.reverse()

    combined_path = forward_path

    if len(reverse_path) > 1:
        combined_path = forward_path + reverse_path[1:]
    
    combined_path_cost = explored_f[crossover] + explored_r[crossover]
    combined_path_node = (combined_path_cost, (combined_path[-1], combined_path))
    return combined_path_node


def reconstruct_path(crossover, pred):

    path = []

    cur = crossover

    while cur != None:
        path.append(cur)
        cur = pred[cur]
    
    path.reverse()
    return path

