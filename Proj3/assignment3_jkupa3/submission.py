import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy͏︍͏︆͏󠄁
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import random
#You are not allowed to use following set of modules from 'pgmpy' Library.͏︍͏︆͏󠄁
#
# pgmpy.sampling.*͏︍͏︆͏󠄁
# pgmpy.factors.*͏︍͏︆͏󠄁
# pgmpy.estimators.*͏︍͏︆͏󠄁
# pgmpy.inference.*͏︍͏︆͏󠄁
# BayesNet.get_state_probability͏︍͏︆͏󠄁

def make_security_system_net():
    """
        Create a Bayes Net representation of the above security system problem. 
        Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
        "D"'. (for the tests to work.)
    """
    bn = BayesianNetwork()
    nodes = ["H","C","M","B","Q","K","D"]
    for node in nodes:
        bn.add_node(node)

    #H
    bn.add_edge("H","Q") #Hiring pro hackers affects the event that Q's database is hacked
    #bn.add_edge("H","D") #Hiring pro hackers affects the event that Spectre succeeds in obtaining the files

    #C
    bn.add_edge("C","Q") #Buying the super-computer Contra affects the event that Q's database is hacked
    #bn.add_edge("C","D") #Buying the super-computer Contra affects the event that Spectre succeeds in obtaining the files

    #M
    bn.add_edge("M","K") #Spectre hiring mercinaries affects the event that M gets kidnapped anf has to give away key
    #bn.add_edge("M","D") #Spectre hiring mercinaries affects the event that Spectre succeeds in obtaining the files

    #B
    bn.add_edge("B","K") #Bond guarding M affects the event that M gets kidnapped anf has to give away key
    #bn.add_edge("B","D") #Bond guarding M affects the event that Spectre succeeds in obtaining the files

    #Q
    bn.add_edge("Q","D") #Q's database being hacked affects the event that Spectre succeeds in obtaining the files

    #K
    bn.add_edge("K","D") #Bond guarding M affects the event that Spectre succeeds in obtaining the files

    #D
    #N/A

    return bn


def set_probability(bayes_net):
    """
        Set probability distribution for each node in the security system.
        Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
        "D"'. (for the tests to work.)
    """
    bn = bayes_net

    probability_list = []

    #Spectre will not be able to find and hire skilled professional hackers (call this false) with a probability of 0.5.
    cpd_h = TabularCPD(variable='H', variable_card=2, values=[[0.5], [0.5]])
    probability_list.append(cpd_h)

    #Spectre will get their hands on Contra (call this true) with a probability of 0.3.
    cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.7], [0.3]])
    probability_list.append(cpd_c)

    #Spectre will be unable to hire the mercenaries (call this false) with a probability of 0.2.
    cpd_m = TabularCPD(variable='M', variable_card=2, values=[[0.2], [0.8]])
    probability_list.append(cpd_m)

    #Since Bond is also assigned to another mission, the probability that he will be protecting M at a given moment (call this true) is just 0.5!
    cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.5], [0.5]])
    probability_list.append(cpd_b)

    #The professional hackers will be able to crack Q’s personal database (call this true) without using Contra with a probability of 0.55. 
    #However, if they get their hands on Contra, they can crack Q’s personal database with a probability of 0.9. 
    #In case Spectre can not hire these professional hackers, their less experienced employees will launch a cyberattack on Q’s personal database. 
    #In this case, Q’s database will remain secure with a probability of 0.75 if Spectre has Contra and with a probability of 0.95 if Spectre does not have Contra.

    #P(Q|H, -C) = 0.55
    #P(Q|H, C) = 0.9
    # H  C  P(Q=True)
    # T  T  0.9
    # T  F  0.55
    # F  T  0.25
    # F  F  0.05
    cdp_hcq = TabularCPD(variable='Q', variable_card=2, values=[[0.95, 0.75, 0.45, 0.1], [0.05, 0.25, 0.55, 0.9]], evidence=['H', 'C'], evidence_card=[2, 2])
    probability_list.append(cdp_hcq)

    #When Bond is protecting M, the probability that M stays safe (call this false) is 0.85 if mercenaries conduct the attack. 
    #Else, when mercenaries are not present, it the probability that M stays safe is as high as 0.99! 
    #However, if M is not accompanied by Bond, M gets kidnapped with a probability of 0.95 and 0.75 respectively, with and without the presence of mercenaries.
    
    # B  M  P(K=False) K is safe
    # T  T  0.85
    # T  F  0.99
    # F  T  0.05
    # F  F  0.25

    cpd_bmk = TabularCPD(variable='K', variable_card=2, values=[[0.25, 0.05, 0.99, 0.85], [0.75, 0.95, 0.01, 0.15]], evidence=['B', 'M'], evidence_card=[2, 2])
    probability_list.append(cpd_bmk)


    #With both the cipher and the key, Spectre can access the “Double-0” files (call this true) with a probability of 0.99! 
    #If Spectre has none of these, then this probability drops down to 0.02! 
    #In case Spectre has just the cipher, the probability that the “Double-0” files remain uncompromised is 0.4. 
    #On the other hand, if Spectre has just the key, then this probability changes to 0.65.

    # Q  K  P(D=True)
    # T  T  0.99
    # T  F  0.60
    # F  T  0.35
    # F  F  0.02

    cpd_qkd = TabularCPD(variable='D', variable_card=2, values=[[0.98, 0.65, 0.40, 0.01], [0.02, 0.35, 0.60, 0.99]], evidence=['Q', 'K'], evidence_card=[2, 2])
    probability_list.append(cpd_qkd)

    bn.add_cpds(*probability_list)
    

    # TODO: set the probability distribution for each node͏︍͏︆͏󠄁
    #raise NotImplementedError    
    return bn


def get_marginal_double0(bayes_net):
    """
        Calculate the marginal probability that Double-0 gets compromised.
    """
    # TODO: finish this function͏︍͏︆͏󠄁
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['D'], joint=False)
    prob = marginal_prob['D'].values
    print(f"Probabilities: {prob}")
    double0_prob = prob[1]
    return double0_prob


def get_conditional_double0_given_no_contra(bayes_net):
    """
        Calculate the conditional probability that Double-0 gets compromised
        given Contra is shut down.
    """
    # TODO: finish this function͏︍͏︆͏󠄁
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'],evidence={'C':0}, joint=False)
    prob = conditional_prob['D'].values
    double0_prob = prob[1]
    return double0_prob


def get_conditional_double0_given_no_contra_and_bond_guarding(bayes_net):
    """
        Calculate the conditional probability that Double-0 gets compromised
        given Contra is shut down and Bond is reassigned to protect M.
    """
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'],evidence={'C':0, 'B': 1}, joint=False)
    prob = conditional_prob['D'].values
    double0_prob = prob[1]
    return double0_prob


def get_game_network():
    """
        Create a Bayes Net representation of the game problem.
        Name the nodes as "A","B","C","AvB","BvC" and "CvA".  
    """
    bn = BayesianNetwork()
    nodes = ["A","B","C","AvB","BvC","CvA"]
    for node in nodes:
        bn.add_node(node)

    bn.add_edge("A","AvB")
    bn.add_edge("B","AvB")
    bn.add_edge("B","BvC")
    bn.add_edge("C","BvC")
    bn.add_edge("C","CvA")
    bn.add_edge("A","CvA")


    prob_list = []


    cpd_A = TabularCPD(variable='A', variable_card=4, values=[[0.15], [0.45], [0.30], [0.10]])
    prob_list.append(cpd_A)
    cpd_B = TabularCPD(variable='B', variable_card=4, values=[[0.15], [0.45], [0.30], [0.10]])
    prob_list.append(cpd_B)
    cpd_C = TabularCPD(variable='C', variable_card=4, values=[[0.15], [0.45], [0.30], [0.10]])
    prob_list.append(cpd_C)
    
    

    cpd_AvB = TabularCPD(variable='AvB', variable_card=3, values=[  #T1:0, T1:0, T1:0, T1:0, T1:1, T1:1, T1:1, T1:1, T1:2, T1:2, T1:2, T1:2, T1:3, T1:3, T1:3, T1:3,
                                                                    #T2:0, T2:1, T2:2, T2:3, T2:0, T2:1, T2:2, T2:3, T2:0, T2:1, T2:2, T2:3, T2:0, T2:1, T2:2, T2:3,
                                                                    [.1,   .2,   .15,  .05,  .6,   .1,   .2,   .15,  .75,   .6,   .1,  .2,   .9,   .75,  .6,   .1], #T1 WINS
                                                                    [.1,   .6,   .75,  .9,   .2,   .1,   .6,   .75,  .15,   .2,   .1,  .6,   .05,  .15,   .2,  .1], #T2 WINS
                                                                    [.8,   .2,   .1,   .05,  .2,   .8,   .2,   .1,   .1,    .2,   .8,  .2,   .05,  .1,  .2,    .8]  #TIE
                                                                ], 
                                                                evidence=['A', 'B'], evidence_card=[4, 4])
    prob_list.append(cpd_AvB)

    cpd_BvC = TabularCPD(variable='BvC', variable_card=3, values=[  #T1:0, T1:0, T1:0, T1:0, T1:1, T1:1, T1:1, T1:1, T1:2, T1:2, T1:2, T1:2, T1:3, T1:3, T1:3, T1:3,
                                                                    #T2:0, T2:1, T2:2, T2:3, T2:0, T2:1, T2:2, T2:3, T2:0, T2:1, T2:2, T2:3, T2:0, T2:1, T2:2, T2:3,
                                                                    [.1,   .2,   .15,  .05,  .6,   .1,   .2,   .15,  .75,   .6,   .1,  .2,   .9,   .75,  .6,   .1], #T1 WINS
                                                                    [.1,   .6,   .75,  .9,   .2,   .1,   .6,   .75,  .15,   .2,   .1,  .6,   .05,  .15,  .2,   .1], #T2 WINS
                                                                    [.8,   .2,   .1,   .05,  .2,   .8,   .2,   .1,   .1,    .2,   .8,  .2,   .05,  .1,   .2,   .8]
                                                                ], 
                                                                evidence=['B', 'C'], evidence_card=[4, 4])
    prob_list.append(cpd_BvC)

    cpd_CvA = TabularCPD(variable='CvA', variable_card=3, values=[  #T1:0, T1:0, T1:0, T1:0, T1:1, T1:1, T1:1, T1:1, T1:2, T1:2, T1:2, T1:2, T1:3, T1:3, T1:3, T1:3,
                                                                    #T2:0, T2:1, T2:2, T2:3, T2:0, T2:1, T2:2, T2:3, T2:0, T2:1, T2:2, T2:3, T2:0, T2:1, T2:2, T2:3,
                                                                    [.1,   .2,   .15,  .05,  .6,   .1,   .2,   .15,  .75,   .6,   .1,  .2,   .9,   .75,  .6,   .1], #T1 WINS
                                                                    [.1,   .6,   .75,  .9,   .2,   .1,   .6,   .75,  .15,   .2,   .1,  .6,   .05,  .15,   .2,  .1], #T2 WINS
                                                                    [.8,   .2,   .1,   .05,  .2,   .8,   .2,   .1,   .1,    .2,   .8,  .2,   .05,  .1,  .2,    .8]  #TIE
                                                                ], 
                                                                evidence=['C', 'A'], evidence_card=[4, 4])
    prob_list.append(cpd_CvA)
    bn.add_cpds(*prob_list)
    # TODO: fill this out͏︍͏︆͏󠄁
    #raise NotImplementedError    
    return bn


def calculate_posterior(bayes_net):
    """
        Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
        Return a list of probabilities corresponding to win, loss and tie likelihood.
    """
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['BvC'], evidence={'AvB': 0, 'CvA': 2}, joint=False) #,evidence={'C':0, 'B': 1}, joint=False)
    post = conditional_prob['BvC'].values
    return post
    # TODO: finish this function͏︍͏︆͏󠄁    
    raise NotImplementedError
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """
        Complete a single iteration of the Gibbs sampling algorithm 
        given a Bayesian network and an initial state value. 
        
        initial_state is a list of length 6 where: 
        index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
        index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
        
        Returns the new state sampled from the probability distribution as a tuple of length 6.
        Return the sample as a tuple. 

        Note: You are allowed to calculate the probabilities for each potential variable
        to be sampled. See README for suggested structure of the sampling process.
    """
    initial_state_tup = tuple(initial_state)    
    # TODO: finish this function͏︍͏︆͏󠄁
    #print(f"Initial State: {initial_state_tup}")

    variable_index = random.choice([0,1,2,4]) # Your chosen variable

    if variable_index == 0: #A
        new_state_A_idx = gs_logic_A(bayes_net, initial_state_tup)
        initial_state[0] = new_state_A_idx
    elif variable_index == 1: #B
        new_state_B_idx = gs_logic_B(bayes_net, initial_state_tup)
        initial_state[1] = new_state_B_idx
    elif variable_index == 2: #C
        new_state_C_idx = gs_logic_C(bayes_net, initial_state_tup)
        initial_state[2] = new_state_C_idx
    # elif variable_index == 3: #AvB
    #     new_state_AvB_idx = gs_logic_AvB(bayes_net, initial_state_tup)
    #    initial_state[3] = new_state_AvB_idx
    elif variable_index == 4: #BvC
        new_state_BvC_idx = gs_logic_BvC(bayes_net, initial_state_tup)
        initial_state[4] = new_state_BvC_idx
    # elif variable_index == 5: #CvA
    #     new_state_CvA_idx = gs_logic_CvA(bayes_net, initial_state_tup)
    #     initial_state[5] = new_state_CvA_idx
    else:
        raise ValueError("Invalid variable index")


    return tuple(initial_state)


def gs_logic_A(bayes_net, initial_state):

    cpd_A = bayes_net.get_cpds("A")
    prob_dist = []

    for a_skill_idx in range(4):
        prob_A = cpd_A.values[a_skill_idx]#[0]
        #AvB
        cpd_AvB = bayes_net.get_cpds("AvB")
        b_skill_idx = initial_state[1]
        AvB_match_outcome_idx = initial_state[3]
        prob_AvB = cpd_AvB.values[AvB_match_outcome_idx][a_skill_idx][b_skill_idx]

        #CvA
        cpd_CvA = bayes_net.get_cpds("CvA")
        c_skill_idx = initial_state[2]
        CvA_match_outcome_idx = initial_state[5]
        prob_CvA = cpd_CvA.values[CvA_match_outcome_idx][c_skill_idx][a_skill_idx]

        comb_prob = prob_A * prob_AvB * prob_CvA
        prob_dist.append(comb_prob)

    prob_sum = sum(prob_dist)
    updated_prob_dist = []
    for p in prob_dist:
        updated_prob_dist.append(p/prob_sum)
    
    new_state = random.choices(range(4), weights=updated_prob_dist)[0]
    return new_state

def gs_logic_B(bayes_net, initial_state):
    cpd_B = bayes_net.get_cpds("B")
    prob_dist = []

    for b_skill_idx in range(4):
        prob_B = cpd_B.values[b_skill_idx]#[0]
        #AvB
        cpd_AvB = bayes_net.get_cpds("AvB")
        a_skill_idx = initial_state[0]
        AvB_match_outcome_idx = initial_state[3]
        prob_AvB = cpd_AvB.values[AvB_match_outcome_idx][a_skill_idx][b_skill_idx]

        #BvC
        cpd_BvC = bayes_net.get_cpds("BvC")
        c_skill_idx = initial_state[2]
        BvC_match_outcome_idx = initial_state[4]
        prob_BvC = cpd_BvC.values[BvC_match_outcome_idx][b_skill_idx][c_skill_idx]

        comb_prob = prob_B * prob_AvB * prob_BvC
        prob_dist.append(comb_prob)

    prob_sum = sum(prob_dist)
    updated_prob_dist = []
    for p in prob_dist:
        updated_prob_dist.append(p/prob_sum)
    
    new_state = random.choices(range(4), weights=updated_prob_dist)[0]
    return new_state


def gs_logic_C(bayes_net, initial_state):
    cpd_C = bayes_net.get_cpds("C")
    prob_dist = []

    for c_skill_idx in range(4):
        prob_C = cpd_C.values[c_skill_idx]#[0]
        #BvC
        cpd_BvC = bayes_net.get_cpds("BvC")
        b_skill_idx = initial_state[1]
        BvC_match_outcome_idx = initial_state[4]
        prob_BvC = cpd_BvC.values[BvC_match_outcome_idx][b_skill_idx][c_skill_idx]

        #CvA
        cpd_CvA = bayes_net.get_cpds("CvA")
        a_skill_idx = initial_state[0]
        CvA_match_outcome_idx = initial_state[5]
        prob_CvA = cpd_CvA.values[CvA_match_outcome_idx][c_skill_idx][a_skill_idx]

        comb_prob = prob_C * prob_BvC * prob_CvA
        prob_dist.append(comb_prob)

    prob_sum = sum(prob_dist)
    updated_prob_dist = []
    for p in prob_dist:
        updated_prob_dist.append(p/prob_sum)
    
    new_state = random.choices(range(4), weights=updated_prob_dist)[0]
    return new_state
        

def gs_logic_AvB(bayes_net, initial_state):
    cpd_AvB = bayes_net.get_cpds("AvB")
    prob_dist = []

    a_skill_idx = initial_state[0]
    b_skill_idx = initial_state[1]

    for AvB_match_outcome_idx in range(3):
        prob_AvB = cpd_AvB.values[AvB_match_outcome_idx][a_skill_idx][b_skill_idx]
        prob_dist.append(prob_AvB)
    
    prob_sum = sum(prob_dist)
    updated_prob_dist = []
    for p in prob_dist:
        updated_prob_dist.append(p/prob_sum)
    
    new_state = random.choices(range(3), weights=updated_prob_dist)[0]
    return new_state


def gs_logic_BvC(bayes_net, initial_state):
    cpd_BvC = bayes_net.get_cpds("BvC")
    prob_dist = []

    b_skill_idx = initial_state[1]
    c_skill_idx = initial_state[2]

    for BvC_match_outcome_idx in range(3):
        prob_BvC = cpd_BvC.values[BvC_match_outcome_idx][b_skill_idx][c_skill_idx]
        prob_dist.append(prob_BvC)
    
    prob_sum = sum(prob_dist)
    updated_prob_dist = []
    for p in prob_dist:
        updated_prob_dist.append(p/prob_sum)
    

    new_state = random.choices(range(3), weights=updated_prob_dist)[0]
    return new_state


def gs_logic_CvA(bayes_net, initial_state):
    cpd_CvA = bayes_net.get_cpds("CvA")
    prob_dist = []

    c_skill_idx = initial_state[2]
    a_skill_idx = initial_state[0]


    print(f"cpd_CvA: {cpd_CvA.values}")

    for CvA_match_outcome_idx in range(3):
        prob_CvA = cpd_CvA.values[CvA_match_outcome_idx][c_skill_idx][a_skill_idx]

        prob_dist.append(prob_CvA)
    
    prob_sum = sum(prob_dist)
    updated_prob_dist = []
    for p in prob_dist:
        updated_prob_dist.append(p/prob_sum)
    
    new_state = random.choices(range(3), weights=updated_prob_dist)[0]
    return new_state


    




def MH_sampler(bayes_net, initial_state):
    """
        Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
        initial_state is a list of length 6 where: 
        index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
        index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
        Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    # A_cpd = bayes_net.get_cpds("A")      
    # AvB_cpd = bayes_net.get_cpds("AvB")
    # match_table = AvB_cpd.values
    # team_table = A_cpd.values
    if initial_state == None:
        initial_skill = [random.randint(0,3) for _ in range(3)] 
        initial_outcome = [0, random.randint(0,2), 2]
        initial_state = initial_skill + initial_outcome
        return tuple(initial_state)

    # sample = tuple(initial_state)  

    # variable_index = random.choice([0,1,2,4]) 

    # candidate_value = 0

    # if 0 <= variable_index <= 2:
    #     candidate_value = random.randint(0,3)
    # elif variable_index == 4:
    #     candidate_value = random.randint(0,2)  
    # else:
    #     raise ValueError("Invalid variable index")

    candidate_state = []
    for _ in range(3):
        candidate_state.append(random.randint(0,3))
    candidate_state.append(0)
    candidate_state.append(random.randint(0,2))
    candidate_state.append(2)
    
    # candidate_state = initial_state.copy()
    # candidate_state[variable_index] = candidate_value

    p_top = probability_MH(bayes_net, candidate_state)
    p_bottom = probability_MH(bayes_net, initial_state)

    # q_top = proposal_MH(initial_state, candidate_state)
    # q_bottom = proposal_MH(candidate_state, initial_state)

    mh = (p_top * 1) / (p_bottom * 1)

    u = random.uniform(0,1)

    accepted = u <= mh

    if accepted:
        return tuple(candidate_state)
    else:
        return tuple(initial_state)


def probability_MH(bayes_net, state):
    cpd_A = bayes_net.get_cpds("A")
    cpd_B = bayes_net.get_cpds("B")
    cpd_C = bayes_net.get_cpds("C")
    cpd_AvB = bayes_net.get_cpds("AvB")
    cpd_BvC = bayes_net.get_cpds("BvC")
    cpd_CvA = bayes_net.get_cpds("CvA")

    idxA, idxB, idxC, idxAvB, idxBvC, idxCvA = state

    prob_A, prob_B, prob_C = cpd_A.values[idxA], cpd_B.values[idxB], cpd_C.values[idxC]
    prob_AvB = cpd_AvB.values[idxAvB][idxA][idxB]
    prob_BvC = cpd_BvC.values[idxBvC][idxB][idxC]
    prob_CvA = cpd_CvA.values[idxCvA][idxC][idxA]

    total_prob = prob_A * prob_B * prob_C * prob_AvB * prob_BvC * prob_CvA
    return total_prob


def proposal_MH(current_state, candidate_state):

    indices_of_interest = [0,1,2,4]
    perterbed_indices = []

    for idx in indices_of_interest:
        if current_state[idx] != candidate_state[idx]:
            perterbed_indices.append(idx)
    
    if len(perterbed_indices) != 1:
        return 0
    
    perterbed_index = perterbed_indices[0]
    
    index_choice_prob = 1/len(perterbed_indices)

    if 0 <= perterbed_index <= 2:
        skill_choice_prob = 1/4
        return index_choice_prob * skill_choice_prob
    elif perterbed_index == 4:
        game_outcome_choice_prob = 1/3
        return index_choice_prob * game_outcome_choice_prob
    else:
        # return 0
        raise ValueError("Invalid variable index")
    




def compare_sampling(bayes_net, initial_state):
    """
        Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge.
    """    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence_dist = [0,0,0] 
    MH_convergence_dist = [0,0,0] 
    N = 100
    delta = .00001

    Gibbs_convergence_dist, Gibbs_count = gibbs_convergence(N, delta)
    MH_convergence_dist, MH_count, MH_rejection_count = mh_convergence(N, delta)

    print(f"Gibbs estimated distribution for BvC: {Gibbs_convergence_dist} in {Gibbs_count} samples")
    print(f"MH estimated distribution for BvC: {MH_convergence_dist} in {MH_count} samples with {MH_rejection_count} rejections")
    posterior = calculate_posterior(bayes_net)
    print("Posterior distribution via exact inference:", posterior)

    return Gibbs_convergence_dist, MH_convergence_dist, Gibbs_count, MH_count, MH_rejection_count

def gibbs_convergence(N, delta):
    """
        Calculate the Gibbs convergence.
    """
    bn = get_game_network()

    initial_skill = [random.randint(0,3) for _ in range(3)] 
    initial_outcome = [0, random.randint(0,2), 2]
    initial_state = initial_skill + initial_outcome
    burn_in = 10000
    outcome_count_BvC = [0, 0, 0]
    
    state = initial_state.copy()
    converge_count = 0
    prev_dist = None
    i = 0

    while True:
        state = list(Gibbs_sampler(bn, state))
        if i < burn_in: 
            print("BURN")
            i += 1
            continue
        
        outcome_BvC = state[4]
        outcome_count_BvC[outcome_BvC] += 1

        cur_dist = []
        cur_samples = i - burn_in + 1
        for count in outcome_count_BvC:
            cur_dist.append(count / cur_samples)
        
        if prev_dist and has_convergence(prev_dist, cur_dist, delta):
            converge_count += 1
        else:
            converge_count = 0
        
        prev_dist = cur_dist
        i += 1

        print(f"Convergence count: {converge_count}")

        if converge_count >= N:
            break





    total_samples = i - burn_in
    gibbs_prob_distribution = prev_dist
    

    return gibbs_prob_distribution, total_samples

def mh_convergence(N, delta):
    """
        Calculate the MH convergence.
    """
    bn = get_game_network()

    initial_skill = [random.randint(0,3) for _ in range(3)] 
    initial_outcome = [0, random.randint(0,2), 2]
    initial_state = initial_skill + initial_outcome
    burn_in = 10000 
    outcome_count_BvC = [0, 0, 0] 
    
    tmp_state = initial_state.copy()
    converge_count = 0
    reject_count = 0
    prev_dist = None
    i = 0

    while True:
        new_state = list(MH_sampler(bn, tmp_state))
        if i < burn_in: 
            print("BURN")
            i += 1
            continue

        if new_state == tmp_state:
            reject_count += 1
        
        outcome_BvC = new_state[4]
        outcome_count_BvC[outcome_BvC] += 1

        cur_dist = []
        cur_samples = i - burn_in + 1
        for count in outcome_count_BvC:
            cur_dist.append(count / cur_samples)
        
        if prev_dist and has_convergence(prev_dist, cur_dist, delta):
            converge_count += 1
        else:
            converge_count = 0
        
        prev_dist = cur_dist
        tmp_state = new_state
        i += 1

        print(f"Convergence count: {converge_count}")

        if converge_count >= N:
            break





    total_samples = i - burn_in
    mh_prob_distribution = prev_dist


    return mh_prob_distribution, total_samples, reject_count

def has_convergence(prev_dist, current_dist, delta):
    total_diff = 0
    for i in range(len(prev_dist)):
        total_diff += abs(prev_dist[i] - current_dist[i])
    return total_diff < delta
    pass

def sampling_question():
    """
        Question about sampling performance.
    """
    # TODO: assign value to choice and factor͏︍͏︆͏󠄁
    # raise NotImplementedError
    choice = 1
    options = ['Gibbs','Metropolis-Hastings']
    factor = 142068 / 117928
    return options[choice], factor


def return_your_name():
    """
        Return your name from this function
    """
    # TODO: finish this function͏︍͏︆͏󠄁
    return "Justin Kupa"
