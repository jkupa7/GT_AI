import numpy as np
import math
from collections import Counter
import time


class DecisionNode:
    """Class to represent a nodes or leaves in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """
        Create a decision node with eval function to select between left and right node
        NOTE In this representation 'True' values for a decision take us to the left.
        This is arbitrary, but testing relies on this implementation.
        Args:
            left (DecisionNode): left child node
            right (DecisionNode): right child node
            decision_function (func): evaluation function to decide left or right
            class_label (value): label for leaf node
        """
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Determine recursively the class of an input array by testing a value
           against a feature's attributes values based on the decision function.

        Args:
            feature: (numpy array(value)): input vector for sample.

        Returns:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice index for data labels.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data contained in the ReadMe.
    It must be built fully starting from the root.
    
    Returns:
        The root node of the decision tree.
    """

    layer2_left = DecisionNode(None, None, None, 0)
    
    layer3_left = DecisionNode(None, None, None, 2)
    layer3_right = DecisionNode(None, None, None, 1)
    func2 = lambda feature: feature[2] <= -0.7045
    layer2_right = DecisionNode(layer3_left, layer3_right, func2, None)

    func1 = lambda feature: feature[1] <= -1.7606
    layer1_right = DecisionNode(layer2_left, layer2_right, func1, None)

    layer1_left = DecisionNode(None, None, None, 0)

    func0 = lambda feature: feature[0] <= .0568
    dt_root = DecisionNode(layer1_left, layer1_right, func0, None)



    return dt_root


def confusion_matrix(true_labels, classifier_output, n_classes=2):
    """Create a confusion matrix to measure classifier performance.
   
    Classifier output vs true labels, which is equal to:
    Predicted  vs  Actual Values.
    
    Output will sum multiclass performance in the example format:
    (Assume the labels are 0,1,2,...n)
                                     |Predicted|
                     
    |A|            0,            1,           2,       .....,      n
    |c|   0:  [[count(0,0),  count(0,1),  count(0,2),  .....,  count(0,n)],
    |t|   1:   [count(1,0),  count(1,1),  count(1,2),  .....,  count(1,n)],
    |u|   2:   [count(2,0),  count(2,1),  count(2,2),  .....,  count(2,n)],'
    |a|   .............,
    |l|   n:   [count(n,0),  count(n,1),  count(n,2),  .....,  count(n,n)]]
    
    'count' function is expressed as 'count(actual label, predicted label)'.
    
    For example, count (0,1) represents the total number of actual label 0 and the predicted label 1;
                 count (3,2) represents the total number of actual label 3 and the predicted label 2.           
    
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
    Returns:
        A two dimensional array representing the confusion matrix.
    """
    c_matrix = np.zeros((n_classes, n_classes), dtype=int)
    true_labels = np.array(true_labels, dtype=int)
    classifier_output = np.array(classifier_output, dtype=int)
    np.add.at(c_matrix, (true_labels, classifier_output), 1)
    return c_matrix


def precision(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the precision of a classifier compared to the correct values.
    In this assignment, precision for label n can be calculated by the formula:
        precision (n) = number of correctly classified label n / number of all predicted label n 
                      = count (n,n) / (count(0, n) + count(1,n) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of precision of each classifier output. 
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [precision (0), precision(1), precision(2), ... precision(n)].
    """
    c_matrix = confusion_matrix(true_labels, classifier_output, n_classes)
    precision_numerator = np.diag(c_matrix)
    precision_denominator = np.sum(c_matrix, axis=0)
    precision = np.divide(precision_numerator, precision_denominator)
    return precision.tolist()



def recall(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the recall of a classifier compared to the correct values.
    In this assignment, recall for label n can be calculated by the formula:
        recall (n) = number of correctly classified label n / number of all true label n 
                   = count (n,n) / (count(n, 0) + count(n,1) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of recall of each classifier output..
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [recall (0), recall (1), recall (2), ... recall (n)].
    """
    c_matrix = confusion_matrix(true_labels, classifier_output, n_classes)
    recall_numerator = np.diag(c_matrix)
    recall_denominator = np.sum(c_matrix, axis=1)
    recall = np.divide(recall_numerator, recall_denominator)
    return recall.tolist()


def accuracy(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """Get the accuracy of a classifier compared to the correct values.
    Balanced Accuracy Weighted:
    -Balanced Accuracy: Sum of the ratios (accurate divided by sum of its row) divided by number of classes.
    -Balanced Accuracy Weighted: Balanced Accuracy with weighting added in the numerator and denominator.

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The accuracy of the classifier output.
    """
    c_matrix = confusion_matrix(true_labels, classifier_output, n_classes)
    accuracy_numerator = np.sum(np.diag(c_matrix))
    accuracy_denominator = np.sum(c_matrix)
    accuracy = accuracy_numerator / accuracy_denominator
    return accuracy


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0, 1, 2, ...
    Returns:
        Floating point number representing the gini impurity.
    """
    values, count = np.unique(class_vector, return_counts=True)
    total = np.sum(count)
    probs = count / total
    probs_squared = probs**2
    gini = 1 - np.sum(probs_squared)
    return gini

def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0, 1, 2....
        current_classes (list(list(int): A list of lists where each list has
            0, 1, 2, ... values).
    Returns:
        Floating point number representing the gini gain.
    """
    # TODO: finish this.͏︍͏︆͏󠄁
    parent_gini = gini_impurity(previous_classes)
    parent_total = len(previous_classes)
    for child in current_classes:
        child_gini = gini_impurity(child)
        child_total = len(child)
        parent_gini -= (child_total / parent_total) * child_gini
    return parent_gini


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=22):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit
        self.feature_indices = None

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        if features.shape[0] == 1:
            #print("Base case of only one row of data left")
            return DecisionNode(None, None, None, classes[0])
        
        if depth >= self.depth_limit:
            #print("Base case of reaching the depth limit")
            mode_class = Counter(classes).most_common(1)[0][0]
            return DecisionNode(None, None, None, mode_class)

        same_y = np.all(classes == classes[0])
        if same_y:
            #print("Base case of all y data points are equivalent")
            return DecisionNode(None, None, None, classes[0])
        
        num_attributes = features.shape[1]
        max_gain = float('-inf')
        max_split = None
        max_idx = None
        parent_gini = gini_impurity(classes)


        for idx in range(num_attributes):

            threshold = np.median(features[:, idx])

            if np.all(features[:, idx] == threshold):
                continue

            left_mask = features[:, idx] <= threshold
            right_mask = features[:, idx] > threshold

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_classes = classes[left_mask]
            right_classes = classes[right_mask]
            left_gini = gini_impurity(left_classes)
            right_gini = gini_impurity(right_classes)
            left_total = len(left_classes)
            right_total = len(right_classes)
            gain = parent_gini - (left_total / len(classes)) * left_gini - (right_total / len(classes)) * right_gini
            if gain > max_gain:
                max_gain = gain
                max_split = threshold
                max_idx = idx
        
        if max_idx is None:
            print("Base case of no more splits possible")
            mode_class = Counter(classes).most_common(1)[0][0]
            return DecisionNode(None, None, None, mode_class)
        
        left_mask = features[:, max_idx] <= max_split
        right_mask = features[:, max_idx] > max_split

        if np.sum(left_mask) == len(classes) or np.sum(right_mask) == len(classes):
            mode_class = Counter(classes).most_common(1)[0][0]
            return DecisionNode(None, None, None, mode_class)

        left_child = self.__build_tree__(features[left_mask], classes[left_mask], depth + 1)
        right_child = self.__build_tree__(features[right_mask], classes[right_mask], depth + 1)

        func = lambda feature, idx=max_idx, split=max_split: feature[idx] <= split

        return DecisionNode(left_child, right_child, func, None)

       
    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = []
        for feature in features:
            if self.root is None:
                print("No tree!")
                return
            class_label = self.root.decide(feature)
            class_labels.append(class_label)
        
        
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """
    
    features, classes = dataset
    num_attributes = features.shape[0]
    indices = np.random.permutation(num_attributes)
    fold_indices = np.array_split(indices, k)
    folds = []
    for i in range(k):
        test_idx = fold_indices[i]
        #leave current i set out for test set
        train_idx = np.concatenate([fold_indices[j] for j in range(k) if j != i])
        train_set = (features[train_idx], classes[train_idx])
        test_set  = (features[test_idx], classes[test_idx])
        folds.append((train_set, test_set))
    return folds


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees=200, depth_limit=5, example_subsample_rate=.1,
                 attr_subsample_rate=.3):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        num_examples = features.shape[0]
        sub_example_size = max(1, int(num_examples * self.example_subsample_rate))

        if np.ndim(features) == 1:
            num_features = 1
        else:   
            num_features = features.shape[1]
        sub_feature_size = max(1, int(num_features * self.attr_subsample_rate))
        
        for tree_idx in range(self.num_trees):
            tree = DecisionTree(self.depth_limit)

            example_indices = np.random.choice(num_examples, size=sub_example_size, replace=True)
            feature_indices = np.random.choice(num_features, size=sub_feature_size, replace=False)
            
            tree.feature_indices = feature_indices
            
            sub_features = features[example_indices][:, feature_indices]
            sub_classes = classes[example_indices]
            tree.fit(sub_features, sub_classes)
            self.trees.append(tree)
        


    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        Returns:
            votes (list(int)): m votes for each element
        """
        votes = None
        for tree in self.trees:
            sub_feature = features[:, tree.feature_indices]
            tree_vote = tree.classify(sub_feature)
            if votes is None:
                votes = np.array(tree_vote)
            else:
                votes = np.vstack((votes, tree_vote))


        mode_votes = []
        num_examples = votes.shape[1]
        for i in range(num_examples):
            example_votes = votes[:, i]
            mode_vote = Counter(example_votes).most_common(1)[0][0]
            mode_votes.append(mode_vote)
        return mode_votes
        
    

class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, n_clf=0, depth_limit=0, example_subsample_rt=0.0, \
                 attr_subsample_rt=0.0, max_boost_cycles=0):
        """Create a boosting class which uses decision trees.
        Initialize and/or add whatever parameters you may need here.
        Args:
             num_clf (int): fixed number of classifiers.
             depth_limit (int): max depth limit of tree.
             attr_subsample_rate (float): percentage of attribute samples.
             example_subsample_rate (float): percentage of example samples.
        """
        self.num_clf = n_clf
        self.depth_limit = depth_limit
        self.example_subsample_rt = example_subsample_rt
        self.attr_subsample_rt=attr_subsample_rt
        self.max_boost_cycles = max_boost_cycles
        # TODO: finish this.͏︍͏︆͏󠄁
        raise NotImplemented()

    def fit(self, features, classes):
        """Build the boosting functions classifiers.
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        # TODO: finish this.͏︍͏︆͏󠄁
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features.
        Predict the labels for each feature in features to its corresponding class
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """
        # TODO: finish this.͏︍͏︆͏󠄁
        raise NotImplemented()


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """
        # TODO: finish this.͏︍͏︆͏󠄁
        
        prod_vect = np.multiply(data, data)
        sum_vect = np.add(prod_vect, data)
        vectorized = sum_vect
        return vectorized

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return (max_sum, max_sum_index)

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        # TODO: finish this.͏︍͏︆͏󠄁
        first_100_data = data[:100, :]
        row_sums = np.sum(first_100_data, axis=1)
        max_row_idx = np.argmax(row_sums)
        max_row_sum = row_sums[max_row_idx]
        return (max_row_sum, max_row_idx)

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        unique_dict = {}
        flattened = data.flatten()
        for item in flattened:
            if item > 0:
                if item in unique_dict:
                    unique_dict[item] += 1
                else:
                    unique_dict[item] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        # TODO: finish this.͏︍͏︆͏󠄁
        flattened = data.flatten()
        positive_mask = flattened > 0
        positive_values = flattened[positive_mask]
        values, counts = np.unique(positive_values, return_counts=True)
        unique_dict = dict(zip(values, counts))
        return unique_dict.items()

    def non_vectorized_glue(self, data, vector, dimension='c'):
        """Element wise array arithmetic with loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        if dimension == 'c' and len(vector) == data.shape[0]:
            non_vectorized = np.ones((data.shape[0],data.shape[1]+1), dtype=float)
            non_vectorized[:, -1] *= vector
        elif dimension == 'r' and len(vector) == data.shape[1]:
            non_vectorized = np.ones((data.shape[0]+1,data.shape[1]), dtype=float)
            non_vectorized[-1, :] *= vector
        else:
            raise ValueError('This parameter must be either c for column or r for row')
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row, col] = data[row, col]
        return non_vectorized

    def vectorized_glue(self, data, vector, dimension='c'):
        """Array arithmetic without loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        vector = np.reshape(vector, (-1, 1) if dimension == 'c' else  (1, -1))
        vectorized = np.concatenate((data, vector), axis=1 if dimension == 'c' else 0)
        return vectorized

    def non_vectorized_mask(self, data, threshold):
        """Element wise array evaluation with loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared.
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        non_vectorized = np.zeros_like(data, dtype=float)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                val = data[row, col]
                if val >= threshold:
                    non_vectorized[row, col] = val
                    continue
                non_vectorized[row, col] = val**2

        return non_vectorized

    def vectorized_mask(self, data, threshold):
        """Array evaluation without loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared. You are required to use a binary mask for this problem
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        vectorized = np.where(data >= threshold, data, data**2)
        return vectorized


def return_your_name():
    
    return 'Justin Kupa'
