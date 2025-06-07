#####################################################
# CS 6601 - Assignment 1͏︍͏︆͏󠄁
# tri_ucs.py͏︍͏︆͏󠄁
#####################################################

# DO NOT ADD OR REMOVE ANY IMPORTS FROM THIS FILE͏︍͏︆͏󠄁
import math
from submission.priority_queue import PriorityQueue

# Credits if any͏︍͏︆͏󠄁
# 1)͏︍͏︆͏󠄁
# 2)͏︍͏︆͏󠄁
# 3)͏︍͏︆͏󠄁

def tridirectional_search(graph, goals) -> list:
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    #debug_goals = [5, 72, 27]
    global_explored = {}
    a, b, c = goals[0], goals[1], goals[2]
    if a == b == c: return []
    ab_str = ".".join(sorted([str(a), str(b)]))
    bc_str = ".".join(sorted([str(b), str(c)]))
    ca_str = ".".join(sorted([str(c), str(a)]))

    partials = {ab_str: PriorityQueue(), bc_str: PriorityQueue(), ca_str: PriorityQueue()}

    a_frontier, b_frontier, c_frontier = PriorityQueue(), PriorityQueue(), PriorityQueue()
    fronts = {a:a_frontier, b:b_frontier, c:c_frontier}
    a_best, b_best, c_best = {a:0}, {b:0}, {c:0}
    best = {a:a_best, b:b_best, c:c_best}
    pred = {x:{x:None} for x in goals}

    a_node = (0, a)
    b_node = (0, b)
    c_node = (0, c)
    fronts[a].append(a_node)
    fronts[b].append(b_node)
    fronts[c].append(c_node)

    best_path = []
    best_path_cost = float('inf')

    goal_set = set(goals)



    while not (a_frontier.size() == 0 and b_frontier.size() == 0 and c_frontier.size() == 0):
        min_front = None
        min_path_cost = float('inf')

        min_goal = None

        for goal in goals:
            cur_front = fronts[goal]
            if cur_front.size() == 0: continue
            cur_path_cost = fronts[goal].top()[0]
            if cur_path_cost < min_path_cost:
                min_front = cur_front
                min_path_cost = cur_path_cost
                min_goal = goal
        
        pop_path_cost, pop_node = min_front.pop()
        if pop_node in global_explored and global_explored[pop_node] != min_goal:


            origin_a = min_goal
            origin_b = global_explored[pop_node]
            collision_set = set([origin_a, origin_b])
            partial_path_cost = best[origin_a][pop_node] + best[origin_b][pop_node]
            collision_str = ".".join(sorted([str(origin_a), str(origin_b)]))
            new_best_partial_path = False
            if partials[collision_str].size() == 0 or partial_path_cost < partials[collision_str].top()[0]:
                new_best_partial_path = True
            partial_path = reconstruct_paths(pop_node, pred, origin_a, origin_b, goals)
            partials[collision_str].append((partial_path_cost, partial_path))
            if new_best_partial_path:
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
                                    shared_node = origin
                                    if shared_node != path1[-1]:
                                        path1.reverse()
                                    if shared_node != path2[0]:
                                        path2.reverse()
                                    best_path = path1 + path2[1:]
                                    break

            continue
        
        global_explored[pop_node] = min_goal 
        sec_min_path_cost = float('inf')
        for key, front in fronts.items():
            if front == min_front or front.size() == 0: continue
            if front.top()[0] < sec_min_path_cost:
                sec_min_path_cost = front.top()[0]

        if pop_path_cost > best_path_cost or pop_path_cost + sec_min_path_cost >= best_path_cost:
            break


        sorted_neighbors = graph.neighbors(pop_node)
        for nei in sorted_neighbors:

            nei_path_cost = pop_path_cost + graph.get_edge_weight(pop_node, nei)

            if nei not in best[min_goal] or nei_path_cost < best[min_goal][nei]:
                best[min_goal][nei] = nei_path_cost
                pred[min_goal][nei] = pop_node
                nei_node = (nei_path_cost, nei)
                min_front.append(nei_node)
    
    return best_path



def reconstruct_paths(crossover, pred, goal_a, goal_b, goals):
    
    
    path_a, path_b = traceback_path(crossover, pred[goal_a]), traceback_path(crossover, pred[goal_b])

    
    path_b.reverse()
    merged_path = path_a + path_b[1:]

  
    return merged_path


def traceback_path(crossover, pred_dict):
    path = []
    cur = crossover
    
    while cur is not None:
        path.append(cur)
        cur = pred_dict[cur]
    path.reverse()
    return path


