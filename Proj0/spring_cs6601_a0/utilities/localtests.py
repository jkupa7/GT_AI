# coding=utf-8

# Author:
# Last Updated: 1/6/2024 by Raymond Jia

import unittest
import random

def print_success_message(msg):
    print(f'UnitTest passed successfully for "{msg}"!')

# Class for Priority Queue testing
class TestPriorityQueue(unittest.TestCase):
    """Test Priority Queue implementation"""
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_append_and_pop(self, PriorityQueue):
        """Test the append and pop functions"""
        queue = PriorityQueue()
        temp_list = []

        for _ in range(10):
            a = random.randint(0, 10000)
            queue.append((a, 'a'))
            temp_list.append(a)

        temp_list = sorted(temp_list)

        for item in temp_list:
            popped = queue.pop()
            self.assertEqual(popped[0], item)
        
        print_success_message(self._testMethodName)

    def test_fifo_property(self, PriorityQueue):
        "Test the fifo property for nodes with same priority"
        queue = PriorityQueue()
        temp_list = [(1, 'b'), (1, 'c'), (1, 'a')]

        for node in temp_list:
            queue.append(node)

        print(f"queue: {queue}") 
        print(f"temp_list: {temp_list}")
        for expected_node in temp_list:
            actual_node = queue.pop()
            self.assertEqual(actual_node[-1], expected_node[-1])

        print_success_message(self._testMethodName)

    def test_fifo_property2(self, PriorityQueue):
        "Test the fifo property for nodes with same priority"
        queue = PriorityQueue()
        temp_list = [(1, 'b'), (-1, 'c'), (100, 'a'), (10.4, 'd'), (1, 'e'), (-1, 'f')]
        exp_list = [(-1, 'c'), (-1, 'f'), (1, 'b'), (1, 'e'), (10.4, 'd'), (100, 'a')]

        for node in temp_list:
            queue.append(node)

        print(f"queue: {queue}") 
        print(f"temp_list: {temp_list}")
        for expected_node in exp_list:
            actual_node = queue.pop()
            self.assertEqual(actual_node[-1], expected_node[-1])

        print_success_message(self._testMethodName)

    # def test_fifo_property3(self, PriorityQueue):
    #     "Test the fifo property for nodes with same priority"
    #     queue = PriorityQueue()
    #     temp_list = [('alice', 'b'), (-1, 'c'), ('john', 'a'), (10.4, 'd'), ('john', 'e'), (-1, 'f'), ('TIM', 'g')]
    #     exp_list = [(-1, 'c'), (-1, 'f'), ('alice', 'b'), ('john', 'e'), ('john', 'a'), (10.4, 'd'), ('TIM', 'g')]

    #     for node in temp_list:
    #         queue.append(node)

    #     print(f"queue: {queue}") 
    #     print(f"temp_list: {temp_list}")
    #     for expected_node in exp_list:
    #         actual_node = queue.pop()
    #         self.assertEqual(actual_node[-1], expected_node[-1])

    #     print_success_message(self._testMethodName)

