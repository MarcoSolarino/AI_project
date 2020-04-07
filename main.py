import copy
import pandas as pd
import math
import numpy
import random
from sklearn.model_selection import train_test_split


def create_data_set(file):
    data = pd.read_csv(file, header=None)
    data.columns = ['attr_{}'.format(int(i)) for i in data.columns]
    data.columns.values[len(data.columns) - 1] = 'class'
    weight = []
    for i in range(len(data.index)):
        weight.append(1)
    data['weight'] = weight
    return data


# elenca i possibili valori degli attributi
def attribute_values(data_set):
    values_matrix = []
    for j in range(len(data_set.columns) - 1):
        values = []
        for i in range(len(data_set.index)):
            if values.__contains__(data_set.iat[i, j]) is False:
                values.append(data_set.iat[i, j])
        values_matrix.append(copy.copy(values))
    return values_matrix


# restituisce la frazione di esempi (con valore dell'attributo noto) con valore attr_value
def compute_fraction(examples, attr_position, attr_value):
    num = 0
    for j in range(len(examples.index)):
        if examples.iat[j, attr_position] == attr_value:
            num += examples.iat[j, len(examples.columns) - 1]
    denom = 0
    for j in range(len(examples.index)):
        if examples.iat[j, attr_position] != '?':
            denom += examples.iat[j, len(examples.columns) - 1]
    return num, denom


def pick_examples(attribute, attr_value, examples):
    k = examples.columns.get_loc(attribute)
    num, denom = compute_fraction(examples, k, attr_value)
    if denom == 0:
        fraction = 0
    else:
        fraction = num / denom
    exs = examples[examples[attribute] == attr_value]
    unknown_examples = examples[examples[attribute] == '?']
    for i in range(len(unknown_examples.index)):
        unknown_examples.iat[i, len(unknown_examples.columns) - 1] *= fraction
    result = exs.append(unknown_examples, ignore_index=True)
    return result


# calcola la probabilità che un esempio abbia per valore dell'attributo indicato da attribute_index value
def prob_attribute_value(attribute_index, value, examples):
    num, denom = compute_fraction(examples, attribute_index, value)
    if denom == 0:
        return 0
    else:
        frac = num / denom  # la probabilità che un esempio abbia value come valore dell'attributo
    for j in range(len(examples.index)):
        if examples.iat[j, attribute_index] == '?':
            num += (examples.iat[j, len(examples.columns) - 1] * frac)
    p = num / examples['weight'].sum()
    return p


def entropy(attribute_index, values, examples):
    e = 0
    for v in values:
        p = prob_attribute_value(attribute_index, v, examples)
        if p != 0:
            e += (p * math.log(p, 2))
    return - e


def remainder(attribute_index, values, class_values, examples):
    r = 0
    for v in values:
        name = examples.columns.values[attribute_index]
        ex_attr_v = examples[(examples[name] == v) | (examples[name] == '?')]
        h = entropy(len(examples.columns) - 2, class_values, ex_attr_v)
        r += prob_attribute_value(attribute_index, v, examples) * h
    return r


def importance(attributes, values_matrix, examples):
    gain_vector = []
    for j in attributes:
        col_index = examples.columns.get_loc(j)
        class_position = len(examples.columns.values) - 2
        h = entropy(class_position, values_matrix[class_position], examples)
        r = remainder(col_index, values_matrix[col_index], values_matrix[class_position], examples)
        gain = h - r
        gain_vector.append(copy.copy(gain))
    find_max = numpy.array(copy.copy(gain_vector))
    result = numpy.amax(find_max)
    for i in range(len(gain_vector)):
        if result == gain_vector[i]:
            return attributes[i]


class Node:
    def __init__(self, pl_vl, attribute):
        self.current_attr = attribute
        self.plurality_class = pl_vl
        self.subtree = []
        self.type = None
        self.arcs = []


def plurality_value(examples):
    most = examples["class"].mode()
    if len(most) == 1:
        return most[0]
    else:
        return random.choice(most)  # tie-breaker


def get_attributes_list(examples):
    attributes = []
    for i in examples.columns.values:
        if (i != "class") and (i != "weight"):
            attributes.append(i)
    return attributes


def same_classification(examples):
    classification = []
    for i in range(len(examples.index)):
        if len(classification) == 0:
            classification.append(examples.iat[i, len(examples.columns) - 2])
        elif examples.iat[i, len(examples.columns) - 2] not in classification:
            return False
    return True


def decision_tree_learning(examples, attrib, values_matrix, pater_examples):
    if len(examples.index) == 0:
        leaf = Node(None, 'class')
        leaf.type = plurality_value(pater_examples)
        return leaf

    if same_classification(examples):
        leaf = Node(None, 'class')
        leaf.type = examples.iat[0, len(examples.columns) - 2]
        return leaf

    if len(attrib) == 0:
        leaf = Node(None, 'class')
        leaf.type = plurality_value(examples)
        return leaf

    attribute = importance(attrib, values_matrix, examples)
    tree = Node(plurality_value(examples), attribute)  # al node viene data la classe più frequente

    col_index = examples.columns.get_loc(attribute)
    values = values_matrix[col_index]  # valori possibili dell'attributo

    new_attributes = copy.copy(attrib)
    new_attributes.remove(attribute)
    for v in values:
        exs = pick_examples(attribute, v, examples)
        tree.arcs.append(v)
        tree.subtree.append(decision_tree_learning(exs, new_attributes, values_matrix, examples))
    return tree


def uniform_deletion(p, data_set):
    for i in range(len(data_set.index)):
        for j in range(len(data_set.columns) - 2):
            x = random.random()
            if x <= p:
                data_set.iat[i, j] = '?'


def classifier(tree, example):
    while tree.current_attr != 'class':
        attribute = tree.current_attr
        value = example[attribute].values[0]
        if value != '?':
            index = tree.arcs.index(value)  # restituisce l'indice dell'arco
            tree = tree.subtree[index]
        else:
            return tree.plurality_class   # ritorna la classe più comune tra gli esempi a quel nodo
    return tree.type


def test_precision(data_set, p):
    data = create_data_set(data_set)
    values_matrix = attribute_values(data)
    attributes = get_attributes_list(data.sample(1))
    uniform_deletion(p, data)
    training_set, test_set = train_test_split(data, test_size=0.2)
    dtree = decision_tree_learning(training_set, attributes, values_matrix, None)
    correct = 0
    for i in range(len(test_set.index)):
        if classifier(dtree, test_set.iloc[[i]]) == test_set.iat[i, len(test_set.columns) - 2]:
            correct += 1
    return correct / len(test_set.index)


# main

probs = [0, 0.1, 0.2, 0.5]

for pb in probs:
    results = []
    for itr in range(20):
        results.append(test_precision("car.data", pb))
    average = 0
    for r in results:
        average += r
    average /= 20
    print("Test su car.data con precisione" + str(pb))
