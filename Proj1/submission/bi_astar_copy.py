#####################################################
# CS 6601 - Assignment 1͏︍͏︆͏󠄁
# bi_astar.py͏︍͏︆͏󠄁
#####################################################

# DO NOT ADD OR REMOVE ANY IMPORTS FROM THIS FILE͏︍͏︆͏󠄁
import math
from submission.priority_queue import PriorityQueue
from submission.astar import euclidean_dist_heuristic

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

def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic) -> list:
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start: Key for the start node.
        goal: Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path via bi-A* as a list from the start to the goal node (including both).
    """

    # TODO: finish this function!͏︍͏︆͏󠄁
    if goal == start:
        return []
    
    f_q = PriorityQueue()
    r_q = PriorityQueue()
    f, r = start, goal
    node_f = (0, (start, [start], 0))
    node_r = (0, (goal, [goal], 0))
    f_q.append(node_f)
    r_q.append(node_r)

    explored_f, explored_r = {start:0}, {goal:0}
    pred_f, pred_r = {start:None}, {goal:None}

    best_path = []
    best_node = (float('inf'), (None, [], float('inf')))

    p_r_t = heuristic(graph, goal, start)

    while (f_q.size() != 0 and r_q.size() != 0):
        f_top, r_top = f_q.top(), r_q.top()
        if f_top[0] + f_top[1][2] + r_top[0] + r_top[1][2] >= best_node[1][2] + p_r_t:
            break

        if f_q.top() < r_q.top():
            best_node = proceed('forward', f_q, explored_f, explored_r, pred_f, pred_r, best_node, graph, start, goal)
            f, best_path = best_node[1][0], best_node[1][1]

            pass

        else: # if r_q <= f_q
            best_node = proceed('reverse', r_q, explored_f, explored_r, pred_f, pred_r, best_node, graph, start, goal)
            r, best_path = best_node[1][0], best_node[1][1]

            pass
    print(best_path)
    return best_path

def proceed(dir, q, explored_f, explored_r, pred_f, pred_r, best_node, graph, start, goal, hueristic=euclidean_dist_heuristic):

    #tmp_path = best_path

    if dir == 'forward':
        f_node = q.pop()
        f_heu, f_tup = f_node
        f, f_path, f_wei = f_tup
        #print(f"Popped F node: {f}")
        explored_f[f] = f_wei

        sorted_neighbors = sorted(graph.neighbors(f))

        for nei in sorted_neighbors:
            #print(f"Nei found in F expansion: {nei}")
            best_wei = f_wei + graph.get_edge_weight(f, nei)
            best_heu = hueristic(graph, nei, goal)
            best_f = best_wei + best_heu
            if best_f <= hueristic(graph, start, goal):
            #if nei not in explored_f or best_wei < explored_f[nei]:
                explored_f[nei] = best_wei
                pred_f[nei] = f
                nei_heu = best_wei + hueristic(graph, nei, goal)
                f_path.append(nei)
                node_nei = (nei_heu, (nei, f_path.copy(), best_wei))
                q.append(node_nei)
                f_path.pop()

                if nei in explored_r:# and nei in pred_r:
                    #print(f"nei in explored_r: {nei}")
                    best_path_cost = explored_f[nei] + explored_r[nei]
                    #union_node = join_nodes(dir, nei, pred_f, pred_r, explored_f, explored_r)
                    if best_path_cost < best_node[1][2]:
                        best_node = join_nodes(dir, nei, pred_f, pred_r, explored_f, explored_r)


    elif dir == 'reverse':
        r_node = q.pop()
        r_heu, r_tup = r_node
        r, r_path, r_wei = r_tup
        #print(f"Popped R node: {r}")
        explored_r[r] = r_wei

        sorted_neighbors = sorted(graph.neighbors(r))

        for nei in sorted_neighbors:
            #print(f"Nei found in R expansion: {nei}")
            best_wei = r_wei + graph.get_edge_weight(r, nei)
            best_heu = hueristic(graph, nei, goal)
            best_f = best_wei + best_heu
            if best_f <= hueristic(graph, start, goal):
            #best_wei = r_wei + graph.get_edge_weight(r, nei)
            #if nei not in explored_r or best_wei < explored_r[nei]:
                explored_r[nei] = best_wei
                pred_r[nei] = r
                nei_heu = best_wei + hueristic(graph, nei, start)
                r_path.append(nei)
                node_nei = (nei_heu, (nei, r_path.copy(), best_wei))
                q.append(node_nei)
                r_path.pop()

                if nei in explored_f:# and nei in pred_f:
                    #print(f"nei in explored_f: {nei}")
                    best_path_cost = explored_f[nei] + explored_r[nei]
                    #union_node = join_nodes(dir, nei, pred_f, pred_r, explored_f, explored_r)
                    if best_path_cost < best_node[1][2]:
                        best_node = join_nodes(dir, nei, pred_f, pred_r, explored_f, explored_r)

    return best_node

def join_nodes(dir, crossover, pred_f, pred_r, explored_f, explored_r):

    forward_path, reverse_path = reconstruct_path(crossover, pred_f), reconstruct_path(crossover, pred_r)

    reverse_path.reverse()

    combined_path = forward_path

    if len(reverse_path) > 1:
        combined_path = forward_path + reverse_path[1:]
    
    combined_path_cost = explored_f[crossover] + explored_r[crossover]
    combined_path_node = (float('inf'), (combined_path[-1], combined_path, combined_path_cost))

    #print(f"combined_path_node: {combined_path_node}")

    return combined_path_node



def reconstruct_path(crossover, pred):

    path = []

    cur = crossover
    #print(f"Pred Dict: {pred}")

    while cur != None:
        path.append(cur)
        cur = pred[cur]
    
    path.reverse()
    return path