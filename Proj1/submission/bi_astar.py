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

    rounded_euc = round(euc_dist, 3)

    return rounded_euc

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
    node_f = (euclidean_dist_heuristic(graph, start, goal), (start, [start], 0))
    node_r = (euclidean_dist_heuristic(graph, goal, start), (goal, [goal], 0))
    f_q.append(node_f)
    r_q.append(node_r)

    explored_f, explored_r = {}, {}
    pred_f, pred_r = {start:None}, {goal:None}

    best_f, best_r = {start:0}, {goal:0}

    best_path = []
    best_node = (float('inf'), (None, [], float('inf')))

    p_r_t = euclidean_dist_heuristic(graph, goal, start)

    while (f_q.size() != 0 and r_q.size() != 0):
        f_top, r_top = f_q.top(), r_q.top()
        f_heu, r_heu = f_top[0], r_top[0]
        f, r = f_top[1][0], r_top[1][0]
        f_wei, r_wei = f_top[1][2], r_top[1][2]
        
        a1 = euclidean_dist_heuristic(graph, f, goal) 
        a2 = euclidean_dist_heuristic(graph, r, goal)

        a = a1 - a2
        
        b = (euclidean_dist_heuristic(graph, r, start) - euclidean_dist_heuristic(graph, f, start))
        

        c = best_f[f] + best_r[r]

        if .5 * (a+b) >= best_node[1][2] - c:
            break

        if len(explored_f) <= len(explored_r):
            best_node = proceed('forward', f_q, explored_f, explored_r, pred_f, pred_r, best_f, best_r, best_node, graph, start, goal)
            f, best_path = best_node[1][0], best_node[1][1]

        else: 
            best_node = proceed('reverse', r_q, explored_f, explored_r, pred_f, pred_r, best_f, best_r, best_node, graph, start, goal)
            r, best_path = best_node[1][0], best_node[1][1]

    return best_path

def proceed(dir, q, explored_f, explored_r, pred_f, pred_r, best_f, best_r, best_node, graph, start, goal, heuristic=euclidean_dist_heuristic):

    if dir == 'forward':
        f_node = q.pop()
        f_heu, f_tup = f_node
        f, f_path, f_wei = f_tup
        if f in explored_f:
            return best_node
        explored_f[f] = f_wei


        sorted_neighbors = sorted(graph.neighbors(f))

        for nei in sorted_neighbors:
            if nei in explored_f: continue 

            nei_edge = graph.get_edge_weight(f, nei)
            nei_wei = nei_edge + f_wei

            if nei not in best_f or nei_wei < best_f[nei]:
                best_f[nei] = nei_wei
                nei_heu = nei_wei + euclidean_dist_heuristic(graph, nei, goal)
                f_path.append(nei)
                nei_node = (nei_heu, (nei, f_path.copy(), nei_wei))
                q.append(nei_node)
                f_path.pop()

                pred_f[nei] = f

                if nei in explored_r:


                    mu = best_f[f] + nei_edge + best_r[nei]
                    mu_node = join_nodes(dir, f, nei, pred_f, pred_r, explored_f, explored_r, best_f, best_r, mu)
                    if mu_node[1][2] < best_node[1][2]:
                        best_node = mu_node

    elif dir == 'reverse':
        
        r_node = q.pop()
        r_heu, r_tup = r_node
        r, r_path, r_wei = r_tup
        if r in explored_r:
            return best_node
        explored_r[r] = r_wei


        sorted_neighbors = sorted(graph.neighbors(r))

        for nei in sorted_neighbors:
            if nei in explored_r: continue

            nei_edge = graph.get_edge_weight(r, nei)
            nei_wei = r_wei + nei_edge
            
            if nei not in best_r or nei_wei < best_r[nei]:
                best_r[nei] = nei_wei
                r_path.append(nei)

                nei_heu = nei_wei + euclidean_dist_heuristic(graph, nei, start)
                nei_node = (nei_heu, (nei, r_path.copy(), nei_wei))
                q.append(nei_node)

                r_path.pop()
                pred_r[nei] = r

                if nei in explored_f:

                    mu = best_f[nei] + nei_edge + best_r[r]
                    mu_node = join_nodes(dir, nei, r, pred_f, pred_r, explored_f, explored_r, best_f, best_r, mu)

                    if mu_node[1][2] < best_node[1][2]:
                        best_node = mu_node

    return best_node


def join_nodes(dir, f_node, r_node, pred_f, pred_r, explored_f, explored_r, best_f, best_r, mu):
    forward_path, reverse_path = reconstruct_path(f_node, pred_f), reconstruct_path(r_node, pred_r)

    reverse_path.reverse()

    
    combined_path = forward_path + reverse_path

    combined_path_node = (mu, (combined_path[-1], combined_path, mu))

    return combined_path_node

    
    




def reconstruct_path(crossover, pred):

    path = []

    cur = crossover

    while cur != None:
        path.append(cur)
        cur = pred[cur]
    
    path.reverse()
    return path