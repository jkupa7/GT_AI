#####################################################
# CS 6601 - Assignment 0
# priority_queue.py
#####################################################

# DO NOT ADD OR REMOVE ANY IMPORTS FROM THIS FILE

class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    You may add extra helper functions within the class if you find them necessary.

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self.count = 0

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        if self.size() == 0:
            return None
        if self.size() == 1:
            return self.queue.pop()[0]
        return_node = self.queue[0][0]
        self.queue[0] = self.queue.pop()
        self.heapify_down(0)
        return return_node
        

    def heapify_down(self, index):
        """
        Move the node at the given index down to its correct position in the heap.
        
        Args:
            index (int): Index of the node to move down.
        """
        node_idx = index
        left_child_idx = 2 * node_idx + 1
        right_child_idx = 2 * node_idx + 2
        if left_child_idx < self.size():
            min_child_idx = left_child_idx
            if right_child_idx < self.size() and (self.queue[right_child_idx][0][0], self.queue[right_child_idx][1]) < (self.queue[left_child_idx][0][0], self.queue[left_child_idx][1]):
                min_child_idx = right_child_idx
            if (self.queue[min_child_idx][0][0], self.queue[min_child_idx][1]) < (self.queue[node_idx][0][0], self.queue[node_idx][1]):
                self.queue[node_idx], self.queue[min_child_idx] = self.queue[min_child_idx], self.queue[node_idx]
                self.heapify_down(min_child_idx)
        



    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """
        
        # We will not test this function, implementation and desired behavior is up to your discretion
        # Some students find that this function is useful for them in Assignment 1
        raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node (tuple): Comparable Object to be added to the priority queue.
            Provided in the form of (int priority, any type payload)
        """

        self.queue.append((node, self.count))
        self.heapify_up(self.size() - 1)
        self.count += 1

    def heapify_up(self, index):
        """
        Move the node at the given index up to its correct position in the heap.

        Args:
            index (int): Index of the node to move up.
        """
        parent_idx = (index - 1) // 2
    
        if index > 0 and (self.queue[index][0][0], self.queue[index][1])  < (self.queue[parent_idx][0][0], self.queue[parent_idx][1]):
            self.queue[index], self.queue[parent_idx] = self.queue[parent_idx], self.queue[index]
            self.heapify_up(parent_idx)
        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]