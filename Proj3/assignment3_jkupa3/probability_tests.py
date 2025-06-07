import unittest
from submission import *
import hashlib
import json
import numpy as np
"""
Contains various local tests for Assignment 3.
"""

class ProbabilityTests(unittest.TestCase):

    #Part 1a͏︍͏︆͏󠄁
    def test_network_setup(self):
        """Test that the power plant network has the proper number of nodes and edges."""
        security_system = make_security_system_net()
        nodes = security_system.nodes()
        self.assertEqual(len(nodes), 7, msg="incorrect number of nodes")
        total_links = security_system.number_of_edges()
        self.assertEqual(total_links, 6, msg="incorrect number of edges between nodes")

    def test_network_node_edge(self):
        """Test the correctness of network nodes and edges."""
        security_system = make_security_system_net()
        nodes = sorted(security_system.nodes())
        edges = sorted(security_system.edges())
        correct_hash = "2e68a90710d552d44fd46a250bc312646bffc6dcdb2a68cafea36bccc8995176"
        def hash_structure(nodes, edges):
            """Hashes the Bayesian network structure (nodes + edges) for secure comparison."""
            structure_str = json.dumps({"nodes": nodes, "edges": edges})
            return hashlib.sha256(structure_str.encode()).hexdigest()
        msg = "Incorrect nodes and/or edges."
        self.assertEqual(correct_hash, hash_structure(nodes, edges), msg)

    #Part 1b͏︍͏︆͏󠄁
    def test_probability_setup(self):
        """Test that all nodes in the power plant network have proper probability distributions.
        Note that all nodes have to be named predictably for tests to run correctly."""
        # test H distribution͏︍͏︆͏󠄁
        security_system = set_probability(make_security_system_net())
        H_node = security_system.get_cpds('H')
        self.assertTrue(H_node is not None, msg='No H node initialized')

        H_dist = H_node.get_values()
        self.assertEqual(len(H_dist), 2, msg='Incorrect H distribution size')
        test_prob = H_dist[0]
        self.assertEqual(round(float(test_prob*100)), 50, msg='Incorrect H distribution')

        # test C distribution͏︍͏︆͏󠄁
        security_system = set_probability(make_security_system_net())
        C_node = security_system.get_cpds('C')
        self.assertTrue(C_node is not None, msg='No C node initialized')

        C_dist = C_node.get_values()
        self.assertEqual(len(C_dist), 2, msg='Incorrect C distribution size')
        test_prob = C_dist[0]
        self.assertEqual(round(float(test_prob*100)), 70, msg='Incorrect C distribution')

        # test M distribution͏︍͏︆͏󠄁
        security_system = set_probability(make_security_system_net())
        M_node = security_system.get_cpds('M')
        self.assertTrue(M_node is not None, msg='No M node initialized')

        M_dist = M_node.get_values()
        self.assertEqual(len(M_dist), 2, msg='Incorrect M distribution size')
        test_prob = M_dist[0]
        self.assertEqual(round(float(test_prob*100)), 20, msg='Incorrect M distribution')

        # test B distribution͏︍͏︆͏󠄁
        security_system = set_probability(make_security_system_net())
        B_node = security_system.get_cpds('B')
        self.assertTrue(B_node is not None, msg='No B node initialized')

        B_dist = B_node.get_values()
        self.assertEqual(len(B_dist), 2, msg='Incorrect B distribution size')
        test_prob = B_dist[0]
        self.assertEqual(round(float(test_prob*100)), 50, msg='Incorrect B distribution')


        # Q distribution͏︍͏︆͏󠄁
        # can't test exact probabilities because͏︍͏︆͏󠄁
        # order of probabilities is not guaranteed͏︍͏︆͏󠄁
        Q_node = security_system.get_cpds('Q')
        self.assertTrue(Q_node is not None, msg='No Q node initialized')
        [cols, rows1, rows2] = Q_node.cardinality
        self.assertEqual(rows1, 2, msg='Incorrect Q distribution size')
        self.assertEqual(rows2, 2, msg='Incorrect Q distribution size')
        self.assertEqual(cols,  2, msg='Incorrect Q distribution size')

        # K distribution͏︍͏︆͏󠄁
        K_node = security_system.get_cpds('K')
        self.assertTrue(K_node is not None, msg='No K node initialized')
        [cols, rows1, rows2] = K_node.cardinality
        self.assertEqual(rows1, 2, msg='Incorrect K distribution size')
        self.assertEqual(rows2, 2, msg='Incorrect K distribution size')
        self.assertEqual(cols,  2, msg='Incorrect K distribution size')

        # D distribution͏︍͏︆͏󠄁
        D_node = security_system.get_cpds('D')
        self.assertTrue(D_node is not None, msg='No D node initialized')
        [cols, rows1, rows2] = D_node.cardinality
        self.assertEqual(rows1, 2, msg='Incorrect D distribution size')
        self.assertEqual(rows2, 2, msg='Incorrect D distribution size')
        self.assertEqual(cols,  2, msg='Incorrect D distribution size')
        try:
            security_system.check_model()
        except:
            self.assertTrue(False, msg='Sum of the probabilities for each state is not equal to 1 or CPDs associated with nodes are not consistent with their parents')

    def test_security_system_probabilities(self):

        def processing(variable, distribution, evidence):
            rounded_distribution = np.round(distribution, 4).tolist()
            structure_str = json.dumps({"variable": variable, "distribution": rounded_distribution, "evidence": evidence})
            return hashlib.sha256(structure_str.encode()).hexdigest()

        message = ''
        # from workspace import probability_solution as pn͏︍͏︆͏󠄁
        try:
           # with NoStd():͏︍͏︆͏󠄁
            security_system = make_security_system_net()
            security_system = set_probability(security_system)
        except:
            message += '1b failed due to error: %s' % (self.PrintException())
        # H͏︍͏︆͏󠄁 
        var = 'H'
        node = security_system.get_cpds(var)
        true_hash = "e09343f69eddf6e7004f37ba72b58a33b7d96f25252a9593277324986eebd7fd"
        dist = node.values
        ev = node.get_evidence()
        self.assertEqual(true_hash, processing(var, dist, ev), msg="Incorrect distribution and/or evidence for node H")

        # C͏︍͏︆͏󠄁 
        var = 'C'
        node = security_system.get_cpds(var)
        true_hash = "7a14e468094a73fc1b460302a163e2b1b2d388c76d7f98e5cb8659e36d424631"
        dist = node.values
        ev = node.get_evidence()
        self.assertEqual(true_hash, processing(var, dist, ev), msg="Incorrect distribution and/or evidence for node C")

        # M͏︍͏︆͏󠄁 
        var = 'M'
        node = security_system.get_cpds(var)
        true_hash = "61f7f43539fd61a9442ff6f5a34adaa20c567ca733e5f17438b18cef59f0ff02"
        dist = node.values
        ev = node.get_evidence()
        self.assertEqual(true_hash, processing(var, dist, ev), msg="Incorrect distribution and/or evidence for node M")

        # B͏︍͏︆͏󠄁
        var = 'B'
        node = security_system.get_cpds(var)
        true_hash = "db6b76ad6f37657651e5ee4c261ed2b368ea128183b346d64273b022a11c1e72"
        dist = node.values
        ev = node.get_evidence()
        self.assertEqual(true_hash, processing(var, dist, ev), msg="Incorrect distribution and/or evidence for node B")

        ###########

        var = 'Q'

        true_hash_1 = "700bcd2d2c5a737b465b416b70c5905979b99b9bae91f761ba983d11cd955643"
        true_hash_2 = "6d09cdc19a22ab15ce9094e6df668cc8baacea5992784f672f0ebf8f769b8324"

        node = security_system.get_cpds(var)

        dist = node.values
        ev = node.get_evidence()

        self.assertIn(processing(var, dist, ev), [true_hash_1, true_hash_2], msg="Incorrect distribution and/or evidence for node Q")

       #############

        var = 'K'

        true_hash_3 = "6b6e0645e05f4c82456eeed3d9aa3659133da74c79c3f2565bd746cd97241f14"
        true_hash_4 = "dd4bf5c028840e4344aa0ad7c161e56b44d266391017409cc064a66208e77148"

        node = security_system.get_cpds(var)

        dist = node.values
        ev = node.get_evidence()

        self.assertIn(processing(var, dist, ev), [true_hash_4, true_hash_3], msg="Incorrect distribution and/or evidence for node K")

        ################

        var = 'D'

        true_hash_5 = "2f7a4f2e2bb78e6e1f841b2c8a78483ed84cb9691947ca76ffd28ffc0c8ce9ff"
        true_hash_6 = "83530d7e2cd7e0008e296adde6d6ad6a5b4ce296762130f9ff0f317a99b1a468"

        node = security_system.get_cpds(var)

        dist = node.values
        ev = node.get_evidence()

        self.assertIn(processing(var, dist, ev), [true_hash_5, true_hash_6], msg="Incorrect distribution and/or evidence for node D")


    def test_bayes_net_check(self):
        """Test that the power plant network has the proper number of nodes and edges."""

        message = ''
        # from workspace import probability_solution as pn͏︍͏︆͏󠄁
        try:
           # with NoStd():͏︍͏︆͏󠄁
            security_system = make_security_system_net()
            security_system = set_probability(security_system)
        except:
            message += '1b failed due to error: %s' % (self.PrintException())
            print(message)

        solver = VariableElimination(security_system)
        conditional_prob = solver.query(variables=['M'], joint=False)
        prob = conditional_prob['M'].values
        double0_prob = prob[0]
        self.assertEqual(round(float(double0_prob*100)),  20, msg=f"Incorrect distribution for D given M, Expected 20%, Actual {(round(float(double0_prob*100)))}")

    # Part 2a Test͏︍͏︆͏󠄁
    def test_games_network(self):
        """Test that the games network has the proper number of nodes and edges."""
        games_net = get_game_network()
        nodes = games_net.nodes()
        self.assertEqual(len(nodes), 6, msg='Incorrect number of nodes')
        total_links = games_net.number_of_edges()
        self.assertEqual(total_links, 6, 'Incorrect number of edges')

        # Now testing that all nodes in the games network have proper probability distributions.͏︍͏︆͏󠄁
        # Note that all nodes have to be named predictably for tests to run correctly.͏︍͏︆͏󠄁

        # First testing team distributions.͏︍͏︆͏󠄁
        # You can check this for all teams i.e. A,B,C (by replacing the first line for 'B','C')͏︍͏︆͏󠄁

        A_node = games_net.get_cpds('B')
        self.assertTrue(A_node is not None, 'Team A node not initialized')
        A_dist = A_node.get_values()
        self.assertEqual(len(A_dist), 4, msg='Incorrect distribution size for Team A')
        test_prob = A_dist[0]
        test_prob2 = A_dist[2]
        self.assertEqual(round(float(test_prob*100)),  15, msg='Incorrect distribution for Team A')
        self.assertEqual(round(float(test_prob2*100)), 30, msg='Incorrect distribution for Team A')

        # Now testing match distributions.͏︍͏︆͏󠄁
        # You can check this for all matches i.e. AvB,BvC,CvA (by replacing the first line)͏︍͏︆͏󠄁
        AvB_node = games_net.get_cpds('BvC')
        self.assertTrue(AvB_node is not None, 'AvB node not initialized')

        AvB_dist = AvB_node.get_values()
        [cols, rows1, rows2] = AvB_node.cardinality
        self.assertEqual(rows1, 4, msg='Incorrect match distribution size')
        self.assertEqual(rows2, 4, msg='Incorrect match distribution size')
        self.assertEqual(cols,  3, msg='Incorrect match distribution size')

        flag1 = True
        flag2 = True
        flag3 = True
        for i in range(0, 4):
            for j in range(0,4):
                x = AvB_dist[:,(i*4)+j]
                if i==j:
                    if x[0]!=x[1]:
                        flag1=False
                if j>i:
                    if not(x[1]>x[0] and x[1]>x[2]):
                        flag2=False
                if j<i:
                    if not (x[0]>x[1] and x[0]>x[2]):
                        flag3=False

        self.assertTrue(flag1, msg='Incorrect match distribution for equal skill levels')
        self.assertTrue(flag2 and flag3, msg='Incorrect match distribution: teams with higher skill levels should have higher win probabilities')

    # Part 2b Test͏︍͏︆͏󠄁
    def test_posterior(self):
        posterior = calculate_posterior(get_game_network())

        self.assertTrue(abs(posterior[0]-0.25)<0.01 and abs(posterior[1]-0.42)<0.01 and abs(posterior[2]-0.31)<0.01, msg='Incorrect posterior calculated')


    def test_gibbs_convergence(self):
        # Build the network and set its CPDs
        bn = get_game_network()
        #bn = set_probability(bn)

        # Generate an initial random state:
        # For teams A,B,C choose skill in [0,3] and for matches AvB, BvC, CvA choose result in [0,2]
        initial_state = [random.randint(0,3) for _ in range(3)] + [0, random.randint(0,2), 2]
        print("Initial state:", initial_state)
        num_iterations = 1000000     # total number of Gibbs iterations
        burn_in = 100000             # ignore these initial samples to let the chain converge
        bvC_counts = [0, 0, 0]      # count occurrences of each outcome for BvC (index 4)
        
        state = initial_state[:]
        
        # Run Gibbs sampling for many iterations
        for i in range(num_iterations):
            # Note: Gibbs_sampler takes the current state and returns a new state (as a tuple)
            state = list(Gibbs_sampler(bn, state))
            if i >= burn_in:
                outcome_BvC = state[4]
                bvC_counts[outcome_BvC] += 1

        total_samples = num_iterations - burn_in
        gibbs_distribution = [count / total_samples for count in bvC_counts]

        print("Gibbs estimated distribution for BvC:", gibbs_distribution)

        # Now compare with the exact posterior from VariableElimination
        posterior = calculate_posterior(bn)
        print("Posterior distribution via exact inference:", posterior)

    def test_gibbs_convergence_pt2(self):
        gibbs_convergence(100, .00001)

    def test_mh_convergence_pt2(self):
        mh_convergence(100, .00001)

    def test_MH_convergence(self):
        # Build the game network
        bn = get_game_network()
        
        # Set evidence: AvB is fixed at 0 and CvA is fixed at 2 (as required by calculate_posterior)
        # For teams A, B, C choose skill in [0,3] and for match BvC choose result in [0,2]
        initial_state = [random.randint(0,3) for _ in range(3)] + [0, random.randint(0,2), 2]
        print("Initial state:", initial_state)
        
        num_iterations = 1000000     # total number of MH iterations
        burn_in = 100000             # burn-in period to allow the chain to stabilize
        mh_counts = [0, 0, 0]        # count occurrences for each outcome of BvC (index 4)
        
        state = initial_state[:]
        for i in range(num_iterations):
            state = list(MH_sampler(bn, state))
            if i >= burn_in:
                mh_counts[state[4]] += 1
        
        total_samples = num_iterations - burn_in
        mh_distribution = [count / total_samples for count in mh_counts]
        
        # Obtain exact posterior distribution (for BvC given AvB=0 and CvA=2)
        exact_posterior = calculate_posterior(bn)
        
        print("MH estimated distribution for BvC:", mh_distribution)
        print("Posterior distribution via exact inference:", exact_posterior)
        
        # Check that the estimated distribution is sufficiently close to the exact posterior.
        # Adjust delta tolerance as needed.
        for i in range(3):
            self.assertAlmostEqual(mh_distribution[i], exact_posterior[i], delta=0.05)

    def test_comparison(self):
        bn = get_game_network()
        compare_sampling(bn, None)

    def test_double0(self):
        bn = make_security_system_net()
        bn = set_probability(bn)
        get_marginal_double0(bn)



if __name__ == '__main__':
    unittest.main()
