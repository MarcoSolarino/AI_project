import copy
import pandas as pd
import math
import numpy
import random
from sklearn.model_selection import train_test_split


def create_data_set(file):
    data = pd.read_csv(file)
    col = len(data.columns)
    for i in range(col):
        data.columns.values[i] = 'attr' + str(i)
    data.columns.values[col - 1] = 'class'
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


def pick_examples(attribute, attr_value, examples):
    num = 0
    k = examples.columns.get_loc(attribute)
    for j in range(len(examples.index)):
        if examples.iat[j, k] == attr_value:
            num += examples.iat[j, len(examples.columns) - 1]
    if num != 0:
        denom = 0
        for j in range(len(examples.index)):
            if examples.iat[j, k] != '?':
                denom += examples.iat[j, len(examples.columns) - 1]
        if denom == 0:
            fraction = 0
        else:
            fraction = num / denom
    else:
        fraction = 0
    exs = examples[examples[attribute] == attr_value]
    unknown_examples = examples[examples[attribute] == '?']
    for i in range(len(unknown_examples.index)):
        unknown_examples.iat[i, len(unknown_examples.columns) - 1] *= fraction
    result = exs.append(unknown_examples, ignore_index=True)
    return result


# calcola la probabilità che un esempio abbia per valore dell'attributo indicato da attribute_index value
def prob_attribute_value(attribute_index, value, examples):
    num = 0
    if len(examples.index) == 0:
        return num
    for j in range(len(examples.index)):  # sommo i pesi degli esempi con valore dell'attributo noto
        if examples.iat[j, attribute_index] == value:
            num += examples.iat[j, len(examples.columns) - 1]
        if num == 0:
            return 0
    denom = 0
    for j in range(len(examples.index)):
        if examples.iat[j, attribute_index] != '?':
            denom += examples.iat[j, len(examples.columns) - 1]
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
            e += - (p * math.log(p, 2))
    return e


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
    tree = Node(plurality_value(examples), attribute)
    col_index = examples.columns.get_loc(attribute)
    values = values_matrix[col_index]
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
            index = tree.arcs.index(value)
            tree = tree.subtree[index]
        else:
            return tree.plurality_class
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

print("Tic Tac Toe")
print("Avvio con probabilità 0: ")
print(test_precision('tic-tac-toe.data', 0))

print("Avvio con probabilità 0.1: ")
print(test_precision('tic-tac-toe.data', 0.1))

print("Avvio con probabilità 0.2: ")
print(test_precision('tic-tac-toe.data', 0.2))

print("Avvio con probabilità 0.5: ")
print(test_precision('tic-tac-toe.data', 0.5))


print("\n" + "Nursery")
print("Avvio con probabilità 0: ")
print(test_precision('nursery.data', 0))

print("Avvio con probabilità 0.1: ")
print(test_precision('nursery.data', 0.1))

print("Avvio con probabilità 0.2: ")
print(test_precision('nursery.data', 0.2))

print("Avvio con probabilità 0.5: ")
print(test_precision('nursery.data', 0.5))


print("\n" + "Car")
print("Avvio con probabilità 0: ")
print(test_precision('car.data', 0))

print("Avvio con probabilità 0.1: ")
print(test_precision('car.data', 0.1))

print("Avvio con probabilità 0.2: ")
print(test_precision('car.data', 0.2))

print("Avvio con probabilità 0.5: ")
print(test_precision('car.data', 0.5))
