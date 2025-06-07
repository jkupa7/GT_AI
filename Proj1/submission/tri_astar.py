#####################################################
# CS 6601 - Assignment 1͏︍͏︆͏󠄁
# tri_astar.py͏︍͏︆͏󠄁
#####################################################

# DO NOT ADD OR REMOVE ANY IMPORTS FROM THIS FILE͏︍͏︆͏󠄁
import math
from submission.priority_queue import PriorityQueue
from submission.astar import euclidean_dist_heuristic

# Credits if any͏︍͏︆͏󠄁
# 1)͏︍͏︆͏󠄁
# 2)͏︍͏︆͏󠄁
# 3)͏︍͏︆͏󠄁

def custom_heuristic(graph, u, v):
    """
        Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
        Args:
            graph (ExplorableGraph): Undirected graph to search.
            u (str): Key for the first node to calculate from.
            v (str): Key for the second node to calculate to.
        Returns:
            Custom heuristic distance between `u` node and `v` node
        """
    pass

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

# def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic) -> list:
#     """
#     Exercise 4: Upgraded Tridirectional Search

#     See README.MD for exercise description.

#     Args:
#         graph (ExplorableGraph): Undirected graph to search.
#         goals (list): Key values for the 3 goals
#         heuristic: Function to determine distance heuristic.
#             Default: euclidean_dist_heuristic.

#     Returns:
#         The best path as a list from one of the goal nodes (including both of
#         the other goal nodes).
#     """
#     #print(f"Goals: {goals}")
#     debug_goals = ['a', 'u', 'l']
#     global_explored = {}
#     start, mid, goal = goals[0], goals[1], goals[2]
#     if start == mid == goal:
#         return []
    
#     f_q = PriorityQueue()
#     r_q = PriorityQueue()
#     m_q = PriorityQueue()
#     f, m, r = start, mid, goal
#     if goals == debug_goals:
#         print(f"Start: {start}")
#         print(f"Mid: {mid}")
#         print(f"Goal: {goal}")
#     node_f = (0, (start, [start], 0))
#     node_m = (0, (mid, [mid], 0))
#     node_r = (0, (goal, [goal], 0))
#     f_q.append(node_f)
#     r_q.append(node_r)
#     m_q.append(node_m)

#     explored_f, explored_r, explored_m = {}, {}, {}
#     pred_f, pred_r, pred_m = {start:None}, {goal:None}, {mid:None}

#     best_f, best_r, best_m = {start:0}, {goal:0}, {mid:0}

#     best_path = []
#     best_f_node, best_m_node, best_r_node = (float('inf'), (None, [], float('inf'))), (float('inf'), (None, [], float('inf'))), (float('inf'), (None, [], float('inf')))


#     best_start, best_mid, best_end = float('inf'), float('inf'), float('inf')

#     while (f_q.size() != 0 and r_q.size() != 0 and m_q.size()):

#         f_top, r_top, m_top = f_q.top(), r_q.top(), m_q.top()
#         f_heu, r_heu, m_heu = f_top[0], r_top[0], m_top[0]
#         f, r, m = f_top[1][0], r_top[1][0], m_top[1][0]
#         f_wei, r_wei, m_wei = f_top[1][2], r_top[1][2], m_top[1][2]

#         #check for break condition
#         f_m_bool, m_r_bool, r_f_bool = False, False, False
#         if best_start == float('inf') or best_mid == float('inf') or best_end == float('inf'):
#             f_m_bool = m_r_bool = r_f_bool = False
#         else:
#             f_m_bool = f_heu + m_heu >= best_start + best_mid
#             m_r_bool = m_heu + r_heu >= best_mid + best_end
#             r_f_bool = r_heu + f_heu >= best_end + best_start

#         if f_m_bool and m_r_bool and r_f_bool:
#             break

#         res = proceed('forward', f_q, explored_f, explored_m, explored_r, pred_f, pred_m, pred_r, best_f, best_m, best_r, best_f_node, best_m_node, best_r_node, graph, start, mid, goal)
#         if goals == debug_goals:
#             print(f"Res F: {res}")
#         (best_f_node), (best_m_node), (best_r_node) = res[0], res[1], res[2]#proceed('reverse', r_q, explored_f, explored_m, explored_r, pred_f, pred_m, pred_r, best_f, best_m, best_r, best_f_node, best_m_node, best_r_node, graph, start, mid, goal)
#         best_start, best_mid, best_end = best_f_node[1][2], best_m_node[1][2], best_r_node[1][2]
#         res = proceed('mid', m_q, explored_f, explored_m, explored_r, pred_f, pred_m, pred_r, best_f, best_m, best_r, best_f_node, best_m_node, best_r_node, graph, start, mid, goal)
#         if goals == debug_goals:
#             print(f"Res M: {res}")
#         (best_f_node), (best_m_node), (best_r_node) = res[0], res[1], res[2]
#         if goals == debug_goals:
#             print(f"Best Start AFTER SPLITTING AFTER MID: {best_start}")
#         best_start, best_mid, best_end = best_f_node[1][2], best_m_node[1][2], best_r_node[1][2]
#         res = proceed('reverse', r_q, explored_f, explored_m, explored_r, pred_f, pred_m, pred_r, best_f, best_m, best_r, best_f_node, best_m_node, best_r_node, graph, start, mid, goal)
#         if goals == debug_goals:
#             print(f"Res R: {res}")
#         (best_f_node), (best_m_node), (best_r_node) = res[0], res[1], res[2]
#         best_start, best_mid, best_end = best_f_node[1][2], best_m_node[1][2], best_r_node[1][2]
#         if goals == debug_goals:
#             print(f"Best Start AFTER SPLITTING AFTER REV: {best_start}")
        

#     #print(goals)
#     best_q = PriorityQueue()
#     best_q.append(best_f_node)
#     best_q.append(best_m_node)
#     best_q.append(best_r_node)

#     partial_node1 = best_q.pop()
#     path1 = partial_node1[1][1]
#     partial_node2 = best_q.pop()
#     path2 = partial_node2[1][1]
#     partial_node3 = best_q.pop()
#     path3 = partial_node3[1][1]

#     for origin in goals:
#         if (origin in path1 and origin in path2):
#             if goals == debug_goals:
#                 print(f"Partial path 1: {path1}")
#                 print(f"Partial path 2: {path2}")
#             shared_node = origin
#             if goals == debug_goals:
#                 print(f"Shared node: {shared_node}")
#             if shared_node != path1[-1]:
#                 path1.reverse()
#             if shared_node != path2[0]:
#                 path2.reverse()
#             best_path = path1 + path2[1:]
#             if goals == debug_goals:
#                 print(f"New best path found: {best_path}")
#             if set(goals).issubset(set(best_path)):
#                 break
#         if (origin in path2 and origin in path3):
#             if goals == debug_goals:
#                 print(f"Partial path 2: {path2}")
#                 print(f"Partial path 3: {path3}")
#             shared_node = origin
#             if goals == debug_goals:
#                 print(f"Shared node: {shared_node}")
#             if shared_node != path2[-1]:
#                 path2.reverse()
#             if shared_node != path3[0]:
#                 path3.reverse()
#             best_path = path2 + path3[1:]
            
#             if goals == debug_goals:
#                 print(f"New best path found: {best_path}")
#                 if set(goals).issubset(set(best_path)): break

#     if goals == debug_goals:
#         print(f"Best Path: {best_path}")
#         print("WE ARE DONNNEE")
#     return best_path



# def proceed(direction, q, explored_f, explored_m, explored_r, pred_f, pred_m, pred_r, best_f, best_m, best_r, best_f_node, best_m_node, best_r_node, graph, start, mid, goal):

#     if direction == 'forward':
#         f_node = q.pop()
#         f_heu, f_tup = f_node
#         f, f_path, f_wei = f_tup
#         if f in explored_f:
#             return best_f_node, best_m_node, best_r_node
#         explored_f[f] = f_wei

#         sorted_neighbors = sorted(graph.neighbors(f))

#         for nei in sorted_neighbors:
#             if nei in explored_f: continue

#             nei_edge = graph.get_edge_weight(f, nei)
#             nei_wei = nei_edge + f_wei

#             if nei not in best_f or nei_wei < best_f[nei]:
#                 best_f[nei] = nei_wei
#                 nei_heu = nei_wei + euclidean_dist_heuristic(graph, nei, goal)
#                 f_path.append(nei)
#                 nei_node = (nei_wei, (nei, f_path.copy(), nei_wei))
#                 q.append(nei_node)
#                 f_path.pop()

                
#                 pred_f[nei] = f

#                 if nei in explored_m:
#                     mu_m = best_f[nei] + best_m[nei] + nei_edge
#                     mu_m_node = join_nodes(f, nei, pred_f, pred_m, explored_f, explored_m, best_f, best_m, mu_m)
#                     if mu_m_node[1][2] < best_m_node[1][2]:
#                         best_m_node = mu_m_node
                
#                 if nei in explored_r:
#                     mu_r = best_f[nei] + best_r[nei] + nei_edge
#                     mu_r_node = join_nodes(f, nei, pred_f, pred_r, explored_f, explored_r, best_f, best_r, mu_r)
#                     if mu_r_node[1][2] < best_r_node[1][2]:
#                         best_r_node = mu_r_node


#     elif direction == 'mid':
#         m_node = q.pop()
#         m_heu, m_tup = m_node
#         m, m_path, m_wei = m_tup
#         if m in explored_m:
#             return best_f_node, best_m_node, best_r_node
#         explored_m[m] = m_wei

#         sorted_neighbors = sorted(graph.neighbors(m))

#         for nei in sorted_neighbors:
#             if nei in explored_m: continue

#             nei_edge = graph.get_edge_weight(m, nei)
#             nei_wei = nei_edge + m_wei

#             if nei not in best_m or nei_wei < best_m[nei]:
#                 best_m[nei] = nei_wei
#                 nei_heu = nei_wei + euclidean_dist_heuristic(graph, nei, goal)
#                 m_path.append(nei)
#                 nei_node = (nei_wei, (nei, m_path.copy(), nei_wei))
#                 q.append(nei_node)
#                 m_path.pop()

#                 pred_m[nei] = m

#                 if nei in explored_f:
#                     mu_f = best_f[nei] + best_m[nei] + nei_edge
#                     # print(f"Node A: {nei}")
#                     # print(f"Node B: {m}")
#                     # print(f"Pred Dict: {pred_f}")
#                     # print(f"Explored Dict: {explored_f}")
#                     # print(f"Best Dict: {best_f}")
#                     mu_f_node = join_nodes(nei, m, pred_f, pred_m, explored_f, explored_m, best_f, best_m, mu_f)
#                     if mu_f_node[1][2] < best_f_node[1][2]:
#                         best_f_node = mu_f_node
                
#                 if nei in explored_r:
#                     mu_r = best_m[nei] + best_r[nei] + nei_edge
#                     mu_r_node = join_nodes(nei, m, pred_r, pred_m, explored_r, explored_m, best_r, best_m, mu_r)
#                     if mu_r_node[1][2] < best_r_node[1][2]:
#                         best_r_node = mu_r_node

#     elif direction == 'reverse':
#         r_node = q.pop()
#         r_heu, r_tup = r_node
#         r, r_path, r_wei = r_tup
#         if r in explored_r:
#             return best_f_node, best_m_node, best_r_node
#         explored_r[r] = r_wei

#         sorted_neighbors = sorted(graph.neighbors(r))

#         for nei in sorted_neighbors:
#             if nei in explored_r: continue

#             nei_edge = graph.get_edge_weight(r, nei)
#             nei_wei = nei_edge + r_wei

#             if nei not in best_r or nei_wei < best_r[nei]:
#                 best_r[nei] = nei_wei
#                 nei_heu = nei_wei + euclidean_dist_heuristic(graph, nei, goal)
#                 r_path.append(nei)
#                 nei_node = (nei_wei, (nei, r_path.copy(), nei_wei))
#                 q.append(nei_node)
#                 r_path.pop()

#                 pred_r[nei] = r

#                 if nei in explored_f:
#                     mu_f = best_f[nei] + best_r[nei] + nei_edge
#                     mu_f_node = join_nodes(nei, r, pred_f, pred_r, explored_f, explored_r, best_f, best_r, mu_f)
#                     if mu_f_node[1][2] < best_f_node[1][2]:
#                         best_f_node = mu_f_node
                
#                 if nei in explored_m:
#                     mu_m = best_m[nei] + best_r[nei] + nei_edge
#                     mu_m_node = join_nodes(nei, r, pred_m, pred_r, explored_m, explored_r, best_m, best_r, mu_m)
#                     if mu_m_node[1][2] < best_m_node[1][2]:
#                         best_m_node = mu_m_node

#     return best_f_node, best_m_node, best_r_node

# def join_nodes(node_a, node_b, pred_a, pred_b, explored_a, explored_b, best_a, best_b, mu):
#     # print(f"Best A: {best_a}")
#     # print(f"Explored A: {explored_a}")
#     # print(f"Pred Dict: {pred_a}")
#     # print(f"Crossover: {node_a}")
#     forward_path, reverse_path= reconstruct_path(node_a, pred_a), reconstruct_path(node_b, pred_b)
#     reverse_path.reverse()

#     combined_path = forward_path + reverse_path

#     combined_path_node = (mu, (combined_path[-1], combined_path, mu))
#     return combined_path_node


# def reconstruct_path(crossover, pred):

    

#     path = []

#     cur = crossover
#     #print(f"Pred Dict: {pred}")

#     while cur != None:
#         path.append(cur)
#         cur = pred[cur]
    
#     path.reverse()
#     return path


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic) -> list:
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    #Path ['m', 'd', 'c', 'r', 'p'] for goal nodes ('p', 'r', 'm') does not match reference. 
    # Path cost was 438 and expected path cost was 430. Expected path is ['m', 'd', 'c', 'p', 'r']"



    #print(f"Goals: {goals}")
    #debug_goals = ['a', 'u', 'l']
    debug_goals = ['p', 'r', 'm']
    #debug_goals = ['z', 's', 'o']
    #global_explored = {}
    a, b, c = goals[0], goals[1], goals[2]
    if a == b == c:
        return []
    
    ab_str = ".".join(sorted([str(a), str(b)]))
    bc_str = ".".join(sorted([str(b), str(c)]))
    ca_str = ".".join(sorted([str(c), str(a)]))

    # print(f"AB: {ab_str}")
    # print(f"BC: {bc_str}")
    # print(f"CA: {ca_str}")

    partials = {ab_str: PriorityQueue(), bc_str: PriorityQueue(), ca_str: PriorityQueue()}

    a_frontier, b_frontier, c_frontier = PriorityQueue(), PriorityQueue(), PriorityQueue()
    fronts = {a:a_frontier, b:b_frontier, c:c_frontier}
    a_best, b_best, c_best = {a:0}, {b:0}, {c:0}
    best = {a:a_best, b:b_best, c:c_best}
    pred = {x:{x:None} for x in goals}

    a_heu = (euclidean_dist_heuristic(graph, a, b) + euclidean_dist_heuristic(graph, a, c)) / 2
    b_heu = (euclidean_dist_heuristic(graph, b, a) + euclidean_dist_heuristic(graph, b, c)) / 2
    c_heu = (euclidean_dist_heuristic(graph, c, a) + euclidean_dist_heuristic(graph, c, b)) / 2

    a_node = (a_heu, (a, b, c, 0))
    b_node = (b_heu, (b, a, c, 0))
    c_node = (c_heu, (c, b, a, 0))
    fronts[a].append(a_node)
    fronts[b].append(b_node)
    fronts[c].append(c_node)

    global_explored = {a:a, b:b, c:c}

    best_path = []
    best_path_cost = float('inf')

    goal_set = set(goals)

    while not (a_frontier.size() == 0 and b_frontier.size() == 0 and c_frontier.size() == 0):

        if goals == debug_goals:
                print(f"Fronts {goals[0]}: {fronts[goals[0]]}")
                print(f"Fronts {goals[1]}: {fronts[goals[1]]}")
                print(f"Fronts {goals[2]}: {fronts[goals[2]]}")
                print(f"Global Explored: {global_explored}")

        min_front = None
        min_frontier_heu = float('inf')
        min_path_cost = float('inf')

        min_goal = None

        for goal in goals:
            cur_front = fronts[goal]
            if cur_front.size() == 0: continue
            cur_frontier_heu = fronts[goal].top()[0]
            if cur_frontier_heu < min_frontier_heu:
                min_front = cur_front
                min_frontier_heu = cur_frontier_heu
                min_goal = goal

        pop_heu, pop_tup = min_front.pop()
        pop_node, pop_opt1, pop_opt2, pop_wei = pop_tup

        if goals == debug_goals:
            print(f"POP NODE: {pop_node}")

        if pop_node in global_explored and global_explored[pop_node] != min_goal:
            if goals == debug_goals:
                print("Collision detected.")
            origin_a = min_goal
            origin_b = global_explored[pop_node]

            #if pop_node in best[origin_a] and pop_wei < best[origin_b][pop_node]:
                # if goals == debug_goals:
                #     print(f"Stealing frontier from {origin_b} to {origin_a}.")
                # steal_frontier(fronts, best, pred, pop_node, origin_a, origin_b, goals)

            collision_set = set([origin_a, origin_b])
            partial_path_cost = best[origin_a][pop_node] + best[origin_b][pop_node]
            collision_str = ".".join(sorted([str(origin_a), str(origin_b)]))
            new_best_partial_path = False
            if partials[collision_str].size() == 0 or partial_path_cost < partials[collision_str].top()[0]:
                if goals == debug_goals:
                    print(f"New best partial path found.")
                new_best_partial_path = True
            if goals == debug_goals:
                print(f"Partials {ab_str}: {partials[ab_str]}")
                print(f"Partials {bc_str}: {partials[bc_str]}")
                print(f"Partials {ca_str}: {partials[ca_str]}")
            partial_path = reconstruct_paths(pop_node, pred, origin_a, origin_b, goals)
            partials[collision_str].append((partial_path_cost, partial_path))
            if goals == debug_goals:
                print(f"Collision detected between {origin_a} and {origin_b} at {pop_node}.")
            if new_best_partial_path:
                if goals == debug_goals:
                    print(f"New best partial path found.")
                missing_goal = list(goal_set - collision_set)[0]
                for col_node in collision_set:
                    path_str = ".".join(sorted([str(col_node), str(missing_goal)]))
                    if partials[path_str].size() != 0:
                        candidate_cost = partials[collision_str].top()[0] + partials[path_str].top()[0]
                        if candidate_cost < best_path_cost:
                            best_path_cost = candidate_cost
                            path1 = partials[collision_str].top()[1] 
                            path2 = partials[path_str].top()[1]
                            for origin in goals:
                                if origin in path1 and origin in path2:
                                    if goals == debug_goals:
                                        print(f"Partial path 1: {path1}")
                                        print(f"Partial path 2: {path2}")
                                    shared_node = origin
                                    if goals == debug_goals:
                                        print(f"Shared node: {shared_node}")
                                    if shared_node != path1[-1]:
                                        path1.reverse()
                                    if shared_node != path2[0]:
                                        path2.reverse()
                                    best_path = path1 + path2[1:]
                                    if goals == debug_goals:
                                        print(f"New best path found: {best_path}")
                                    break

            
            continue


        # if pop_node in global_explored and pop_node < best[min_goal][pop_node]:
        #     steal_frontier(fronts, best, pred, min_goal, global_explored[pop_node])


        global_explored[pop_node] = min_goal


        # origin_node = pop_node
        # goal_node = pop_goal
        

        if euclidean_dist_heuristic(graph, pop_node, pop_opt1) < euclidean_dist_heuristic(graph, pop_node, pop_opt2):
            goal_node = pop_opt1
            goal = pop_opt1
        else:
            goal_node = pop_opt2
            goal = pop_opt2


        if fronts[goal_node].size() != 0:

            goal_node = fronts[goal_node].top()[1][0]

        a1 = euclidean_dist_heuristic(graph, pop_node, goal_node) 
        a2 = euclidean_dist_heuristic(graph, goal_node, pop_node)
        b1 = euclidean_dist_heuristic(graph, pop_node, goal_node) 
        b2 = euclidean_dist_heuristic(graph, goal_node, pop_node)

        a = a1 - a2
        b = b1 - b2
        c = best[min_goal][pop_node] + best[goal][goal_node]

        # if .5 * (a+b) >= best_path_cost - c:
        #     break

        # if best_path_cost < pop_wei:
        #     break

        sec_min_path_cost = float('inf')
        for key, front in fronts.items():
            if front == min_front or front.size() == 0: continue
            #check this to not redundently add same soruce node 
            if front.top()[0] < sec_min_path_cost:
                sec_min_path_cost = front.top()[0]

        if pop_wei > best_path_cost or pop_wei + sec_min_path_cost >= best_path_cost:
            if goals == debug_goals:
                print(f"pop_wei: {pop_wei}")
                print(f"Fronts {goals[0]}: {fronts[goals[0]]}")
                print(f"Fronts {goals[1]}: {fronts[goals[1]]}")
                print(f"Fronts {goals[2]}: {fronts[goals[2]]}")
                print("BREAK CONDITION INEQUALITY")
                print(f"Best {goals[0]}: {best[goals[0]]}")
                print(f"Best {goals[1]}: {best[goals[1]]}")
                print(f"Best {goals[2]}: {best[goals[2]]}")
                print(f"Global Explored: {global_explored}")
                print(f"Pred {goals[0]}: {pred[goals[0]]}")
                print(f"Pred {goals[1]}: {pred[goals[1]]}")
                print(f"Pred {goals[2]}: {pred[goals[2]]}")
            break

        sorted_neighbors = sorted(graph.neighbors(pop_node))


        for nei in sorted_neighbors:
            # if goals == debug_goals:
            #     print(f"Neighbor: {nei}")
            #     print(f"Best {min_goal}: {best[min_goal]}")
            best_wei = pop_wei + graph.get_edge_weight(pop_node, nei)
            if nei not in best[min_goal] or best_wei < best[min_goal][nei]:
                best[min_goal][nei] = best_wei
                pred[min_goal][nei] = pop_node
                nei_heu = best_wei + ((heuristic(graph, nei, pop_opt1) + heuristic(graph, nei, pop_opt2))/2)
                min_front.append((nei_heu, (nei, pop_opt1, pop_opt2, best_wei)))
            if goals == debug_goals:
                print(f"Neighbor: {nei}")
                print(f"Best {min_goal}: {best[min_goal]}")
    if goals == debug_goals:
        print(f"Best Path: {best_path}")
        print("WE ARE DONNNEE")
    return best_path

    # TODO: finish this function͏︍͏︆͏󠄁
    raise NotImplementedError

#"Path ['s', 'a', 'z', 'o'] for goal nodes ('z', 's', 'o') does not match reference. 
# Path cost was 286 and expected path cost was 222. Expected path is ['s', 'o', 'z']",


def reconstruct_paths(crossover, pred, goal_a, goal_b, goals):
    """
    Args:
        crossover (str): The node all three searches have reached.
        pred (dict): A dict of dicts: pred[origin][node] = predecessor.
        goals (list): The three goal nodes (origins).

    Returns:
        list: A merged path that includes all three goal nodes.
    """
    # Reconstruct each origin->crossover path
    if goals == ['p', 'r', 'm']:
        print(f"Crossover: {crossover}")
        print(f"Pred: {pred}")
        print(f"Goal A: {goal_a}")
        print(f"Goal B: {goal_b}")
        # print(f"Path A: {path_a}")
        # print(f"Path B: {path_b}")
    path_a, path_b = traceback_path(crossover, pred[goal_a], goals), traceback_path(crossover, pred[goal_b], goals)

    if goals == ['p', 'r', 'm']:
        # print(f"Crossover: {crossover}")
        # print(f"Pred: {pred}")
        print(f"Path A: {path_a}")
        print(f"Path B: {path_b}")
    
    # Merge the two paths
    path_b.reverse()
    merged_path = path_a + path_b[1:]

    if goals == ['p', 'r', 'm']:
        print(f"Merged path: {merged_path}")

    return merged_path


def traceback_path(crossover, pred_dict, goals):
    path = []
    cur = crossover
    #if goals == ['a', 'u', 'l']:
        # print(f"Crossover: {crossover}")
        # print(f"Pred: {pred}")
        # print(f"Pred Dict: {pred_dict}")
        # l = "l"
        # if l in pred_dict:
        #     print(f"Pred Dict L: {pred_dict[l]}")
    while cur is not None:
        # if goals == ['a', 'u', 'l']:
        #     print(f"Current Node: {cur}")
        path.append(cur)
        cur = pred_dict[cur]
    path.reverse()
    return path


def steal_frontier(fronts, best, pred, collision, thief_origin, lick_origin, goals):

    thief_front, thief_best, thief_pred = fronts[thief_origin], best[thief_origin], pred[thief_origin]
    lick_front, lick_best, lick_pred = fronts[lick_origin], best[lick_origin], pred[lick_origin]

    if goals == ['a', 'u', 'l'] and thief_origin == 'u' and lick_origin == 'l':
        print(f"Collision detected: {collision}")
        print(f"Lick Front: {lick_front}")
        print(f"Thief Pred: {thief_pred}")
        print(f"Lick Pred: {lick_pred}")

    while lick_front.size() != 0:
        cost, (lick_node, opt1, opt2, lick_wei) = lick_front.pop()

        with_lick_path = lick_best[lick_node]
        without_lick_path = lick_best[lick_node] - lick_best[collision]
        with_thief_path = thief_best[collision] + without_lick_path

        if lick_node == thief_origin: continue

        # if goals == ['a', 'u', 'l'] and lick_node == 'c':
        #     print(f"Lick Node: {lick_node}")
        #     print(f"Lick Node Pred: {lick_pred[lick_node]}")

        if goals == ['z', 's', 'o']and thief_origin == 'u' and lick_origin == 'l':
            
                print(f"Lick Node: {lick_node}")
                print(f"With Lick Path: {with_lick_path}")
                print(f"With Thief Path: {with_thief_path}")
                print(f"Without Lick Path: {without_lick_path}")


        # if with_thief_path < with_lick_path:
            
        thief_best[lick_node] = with_thief_path
        thief_pred[lick_node] = lick_pred[lick_node]
        thief_heu = cost - lick_best[lick_node] + thief_best[lick_node]
        thief_front.append((thief_heu, (lick_node, opt1, opt2, with_thief_path)))
        # else:
        #     thief_best[lick_node] = lick_best[lick_node]
        #     thief_pred[lick_node] = lick_pred[lick_node]
        #     thief_front.append((cost, (lick_node, opt1, opt2, lick_wei)))


        # if lick_node not in thief_best :
        #     thief_best[node] = cost
        #     thief_pred[node] = lick_pred[node]
        #     new_heu = cost
        #     thief_front.append((cost, (node, opt1, opt2, g_val)))
        # elif thief_best[lick_node] < cost:

    # if goals == ['z', 's', 'o'] and thief_origin == 'u' and lick_origin == 'l':
    #     print(f"Thief Pred: {thief_pred}")
    #     print(f"Lick Pred: {lick_pred}")
    fronts[lick_origin].clear()






# def tridirectional_upgraded2(graph, goals, heuristic=euclidean_dist_heuristic) -> list:
#     debug_goals = ['a', 'u', 'l']
#     global_explored = {}
#     a, b, c = goals[0], goals[1], goals[2]
#     if a == b == c:
#         return []
    
#     ab_str = ".".join(sorted([str(a), str(b)]))
#     bc_str = ".".join(sorted([str(b), str(c)]))
#     ca_str = ".".join(sorted([str(c), str(a)]))

#     # print(f"AB: {ab_str}")
#     # print(f"BC: {bc_str}")
#     # print(f"CA: {ca_str}")

#     partials = {ab_str: PriorityQueue(), bc_str: PriorityQueue(), ca_str: PriorityQueue()}

#     a_frontier, b_frontier, c_frontier = PriorityQueue(), PriorityQueue(), PriorityQueue()
#     fronts = {a:a_frontier, b:b_frontier, c:c_frontier}
#     a_best, b_best, c_best = {a:0}, {b:0}, {c:0}
#     best = {a:a_best, b:b_best, c:c_best}
#     pred = {x:{x:None} for x in goals}

#     # a_heu = min(euclidean_dist_heuristic(graph, a, b), euclidean_dist_heuristic(graph, a, c))
#     # b_heu = min(euclidean_dist_heuristic(graph, b, a), euclidean_dist_heuristic(graph, b, c))
#     # c_heu = min(euclidean_dist_heuristic(graph, c, a), euclidean_dist_heuristic(graph, c, b))

#     # a_node = (a_heu, (a, a, b, 0))
#     # b_node = (b_heu, (b, b, c, 0))
#     # c_node = (c_heu, (c, c, a, 0))

#     a_node = (0, (a, b, c, 0))
#     b_node = (0, (b, a, c, 0))
#     c_node = (0, (c, b, a, 0))
#     fronts[a].append(a_node)
#     fronts[b].append(b_node)
#     fronts[c].append(c_node)

#     best_path = []
#     best_path_cost = float('inf')

#     goal_set = set(goals)

#     while not (a_frontier.size() == 0 and b_frontier.size() == 0 and c_frontier.size() == 0):

#         min_front = None
#         min_frontier_size = float('inf')
#         min_path_cost = float('inf')

#         min_goal = None

#         for goal in goals:
#             cur_front = fronts[goal]
#             if cur_front.size() == 0: continue
#             cur_frontier_size = fronts[goal].size()
#             if cur_frontier_size < min_frontier_size:
#                 min_front = cur_front
#                 min_frontier_size = cur_frontier_size
#                 min_goal = goal

#         pop_heu, pop_tup = min_front.pop()
#         pop_node, pop_opt1, pop_opt2, pop_wei = pop_tup

#         # concede_exploration = False
#         # for origin in goals:
#         #     if origin == min_goal: continue

#         #     if pop_node in best[origin] and best[origin][pop_node]:
#         #         concede_exploration = True
        
#         # if concede_exploration:
#         #     print(f"Conceding exploration at {pop_node}.")
#         #     continue

#         if pop_node in global_explored and global_explored[pop_node] != min_goal:
#             origin_a = min_goal
#             origin_b = global_explored[pop_node]

#             collision_set = set([origin_a, origin_b])
#             partial_path_cost = pop_wei + best[origin_a][pop_node] + best[origin_b][pop_node]
#             collision_str = ".".join(sorted([str(origin_a), str(origin_b)]))
#             new_best_partial_path = False
#             if partials[collision_str].size() == 0 or partial_path_cost < partials[collision_str].top()[0]:
#                 new_best_partial_path = True
#             partial_path = reconstruct_paths(pop_node, pred, origin_a, origin_b, goals)
#             partials[collision_str].append((partial_path_cost, partial_path))
#             if goals == debug_goals:
#                 print(f"Collision detected between {origin_a} and {origin_b} at {pop_node}.")
#             if new_best_partial_path:
#                 if goals == debug_goals:
#                     print(f"New best partial path found.")
#                 missing_goal = list(goal_set - collision_set)[0]
#                 for col_node in collision_set:
#                     path_str = ".".join(sorted([str(col_node), str(missing_goal)]))
#                     if partials[path_str].size() != 0:
#                         candidate_cost = partials[collision_str].top()[0] + partials[path_str].top()[0]
#                         if candidate_cost < best_path_cost:
#                             best_path_cost = candidate_cost
#                             path1 = partials[collision_str].top()[1] 
#                             path2 = partials[path_str].top()[1]
#                             for origin in goals:
#                                 if origin in path1 and origin in path2:
#                                     if goals == debug_goals:
#                                         print(f"Partial path 1: {path1}")
#                                         print(f"Partial path 2: {path2}")
#                                     shared_node = origin
#                                     if goals == debug_goals:
#                                         print(f"Shared node: {shared_node}")
#                                     if shared_node != path1[-1]:
#                                         path1.reverse()
#                                     if shared_node != path2[0]:
#                                         path2.reverse()
#                                     best_path = path1 + path2[1:]
#                                     if goals == debug_goals:
#                                         print(f"New best path found: {best_path}")
#                                     break

#             continue

#         # global_explored[pop_node] = min_goal

#         if euclidean_dist_heuristic(graph, pop_node, pop_opt1) < euclidean_dist_heuristic(graph, pop_node, pop_opt2):
#             goal_node = pop_opt1
#             goal = pop_opt1
#         else:
#             goal_node = pop_opt2
#             goal = pop_opt2


#         # if fronts[goal_node].size() != 0:

#         #     goal_node = fronts[goal_node].top()[1][0]

#         a1 = euclidean_dist_heuristic(graph, pop_node, goal_node) 
#         a2 = euclidean_dist_heuristic(graph, goal_node, pop_node)
#         b1 = euclidean_dist_heuristic(graph, pop_node, goal_node) 
#         b2 = euclidean_dist_heuristic(graph, goal_node, pop_node)

#         a = a1 - a2
#         b = b1 - b2
#         c = best[min_goal][pop_node] + best[goal][goal_node]

#         # origin_node = pop_node
#         # goal_node = pop_goal
#         # if fronts[goal_node].size() != 0:
#         #     goal_node = fronts[goal_node].top()[1][0]

#         # a1 = euclidean_dist_heuristic(graph, pop_node, goal_node) 
#         # a2 = euclidean_dist_heuristic(graph, goal_node, pop_node)
#         # b1 = euclidean_dist_heuristic(graph, pop_node, goal_node) 
#         # b2 = euclidean_dist_heuristic(graph, goal_node, pop_node)

#         # a = a1 - a2
#         # b = b1 - b2
#         # c = best[pop_origin][pop_node] + best[pop_goal][goal_node]

#         if .5 * (a+b) >= best_path_cost - c:
#             break

#         sorted_neighbors = sorted(graph.neighbors(pop_node))

#         for nei in sorted_neighbors:
#             nei_edge = graph.get_edge_weight(pop_node, nei)
#             best_wei = pop_wei + nei_edge
#             if nei not in best[min_goal] or best_wei < best[min_goal][nei]:
#                 best[min_goal][nei] = best_wei
#                 pred[min_goal][nei] = pop_node
#                 nei_heu = best_wei + heuristic(graph, nei, goal)
#                 min_front.append((nei_heu, (nei, pop_opt1, pop_opt2, best_wei)))

#                 if nei in best[goal]:

#                     mu = best[min_goal][nei] + nei_edge + best[goal][nei]
#                     #print(f"Mu: {mu}")
#                     if mu < best_path_cost:
#                         best_path_cost = mu
#                         best_path = reconstruct_paths(nei, pred, min_goal, goal, goals)

#     #print(f"Best path: {best_path}")
#     return best_path

