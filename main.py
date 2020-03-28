import copy
import pandas as pd
import math
import numpy
import random

# pd.options.mode.chained_assignment = None

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
            if values.__contains__(data_set.iloc[i, j]) is False:
                values.append(data_set.iloc[i, j])
        values_matrix.append(copy.copy(values))
    return values_matrix


def pick_examples(attribute, attr_value, examples):
    total = len(examples.index)
    exs = examples[examples[attribute] == attr_value]
    fraction = len(exs.index) / total
    unknown_examples = examples[examples[attribute] == '?']
    for i in range(len(unknown_examples.index)):
        # unknown_examples.iloc[i, len(unknown_examples.columns) - 1] = fraction
        unknown_examples.iat[i, len(unknown_examples.columns) - 1] = fraction
    result = exs.append(unknown_examples, ignore_index=True)
    return result


# calcola la probabilità che un esempio abbia per valore dell'attributo indicato da attribute_index value
def prob_attribute_value(attribute_index, value, examples):
    prob = 0
    if len(examples.index) == 0:
        return prob
    for j in range(len(examples.index)):  # sommo i pesi degli esempi con valore dell'attributo noto
        if examples.iloc[j, attribute_index] == value:
            prob += examples.iloc[j, len(examples.columns) - 1]
    frac = copy.copy(prob) / len(examples.index)  # la probabilità che un esempio abbia value come valore dell'attributo
    for j in range(len(examples.index)):
        if examples.iloc[j, attribute_index] == '?':
            prob += examples.iloc[j, len(examples.columns) - 1] * frac
    p = prob/len(examples.index)
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
        r += prob_attribute_value(attribute_index, v, examples) \
             * entropy(len(examples.columns) - 2, class_values, ex_attr_v)
    return r


def importance(attributes, values_matrix, examples):
    gain_vector = []
    for j in attributes:
        col_index = examples.columns.get_loc(j)
        class_position = len(examples.columns.values) - 2
        gain = entropy(class_position, values_matrix[class_position], examples) \
                - remainder(col_index, values_matrix[col_index], values_matrix[class_position], examples)
        gain_vector.append(copy.copy(gain))
    find_max = numpy.array(copy.copy(gain_vector))
    result = numpy.amax(find_max)
    for i in range(len(gain_vector)):
        if result == gain_vector[i]:
            return attributes[i]


class Node:
    def __init__(self, examples, attribute):
        self.examples = examples
        self.current_attr = attribute
        self.subtree = []
        self.type = None
        self.arcs = []


def plurality_value(examples):
    max = examples["class"].mode()
    if len(max) == 1:
        return max[0]
    else:
        return random.choice(max)  # tie-breaker


def get_attributes_list(examples):
    list = []
    for i in examples.columns.values:
        if (i != "class") and (i != "weight"):
            list.append(i)
    return list


def same_classification(examples):
    classification = []
    for i in range(len(examples.index)):
        if len(classification) == 0:
            classification.append(examples.iloc[i, len(examples.columns) - 2])
        elif examples.iloc[i, len(examples.columns) - 2] not in classification:
            return False
    return True


def decision_tree_learning(examples, attrib, values_matrix, pater_examples):
    if len(examples.index) == 0:
        leaf = Node(None, 'class')
        leaf.type = plurality_value(pater_examples)
        return leaf

    if same_classification(examples):
        leaf = Node(None, 'class')
        leaf.type = examples.iloc[0, len(examples.columns) - 2]
        return leaf

    if len(attrib) == 0:
        leaf = Node(None, 'class')
        leaf.type = plurality_value(examples)
        return leaf

    attribute = importance(attrib, values_matrix, examples)
    tree = Node(examples, attribute)
    col_index = examples.columns.get_loc(attribute)
    values = values_matrix[col_index]

    for v in values:
        exs = pick_examples(attribute, v, examples)
        new_attributes = copy.copy(attrib)
        new_attributes.remove(attribute)
        tree.arcs.append(v)
        tree.subtree.append(decision_tree_learning(exs, new_attributes, values_matrix, examples))
    return tree


def uniform_deletion(p, data_set):
    for i in range(len(data_set.index)):
        for j in range(len(data_set.columns) - 2):
            x = random.random()
            if x <= p:
                # data_set.iloc[i, j] = '?'
                data_set.iat[i, j] = '?'


def classifier(tree, example):
    while tree.current_attr != 'class':
        attribute = tree.current_attr
        value = example[attribute].values[0]
        if value != '?':
            index = tree.arcs.index(value)
            tree = tree.subtree[index]
        else:
            return plurality_value(tree.examples)
    return tree.type


def test_precision(data_set, p):
    data = create_data_set(data_set)
    values_matrix = attribute_values(data)
    uniform_deletion(p, data)
    training_set = data.sample(frac=0.5)
    attributes = get_attributes_list(training_set)
    dtree = decision_tree_learning(training_set, attributes,  values_matrix, None)

    success = 0
    for i in range(200):

        sample = data.sample(1)

        if classifier(dtree, sample) == sample.iloc[0, len(sample.columns) - 2]:
            success += 1

    return success / 200


# main

for i in range(0, 6):

    print('p = ' + str(i / 10) + ': ' + str(test_precision('car.data', i / 10) * 100) + '%')
