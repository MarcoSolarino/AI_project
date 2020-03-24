import copy
import pandas as pd
import math
import numpy
import random


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


data = create_data_set("kr-vs-kp.data")


# elenca i possibili valori di un attributo
def attribute_values(attribute_index, data):
    values = []
    for i in range(len(data.index)):
        #  index = examples.columns.get_loc(attribute_index)
        if values.__contains__(data.iloc[i][attribute_index]) is False:
            values.append(data.iloc[i][attribute_index])
    return values


def split_data_set(attribute_index, examples):
    df_tuple = dict(tuple(examples.index.groupby(attribute_index)))
    return df_tuple


# TODO aggiornamento dei pesi per gli attributi mancanti
# calcola la probabilità che un esempio abbia per valore dell'attributo indicato da attribute_index value
def prob_attribute_value(attribute_index, value, examples):
    prob = 0
    for j in range(len(examples.index)):  # sommo i pesi degli esempi con valore dell'attributo noto
        if examples.iloc[j][attribute_index] == value:
            prob += examples.iloc[j][len(examples.columns) - 1]
    frac = copy.copy(prob) / len(examples.index)  # la probabilità che un esempio abbia value come valore dell'attributo
    for j in range(len(examples.index)):
        if examples.iloc[j][attribute_index] == '?':
            prob += examples.iloc[j][len(examples.columns) - 1] * frac
    return prob/len(examples.index)


def entropy(attribute_index, examples):
    values = attribute_values(attribute_index, examples)
    e = 0
    for v in values:
        p = prob_attribute_value(attribute_index, v, examples)
        e += - (p * math.log(p, 2))
    return e


def remainder(attribute_index, examples):
    values = attribute_values(attribute_index, examples)
    r = 0
    for v in values:
        name = examples.columns.values[attribute_index]
        ex_attr_v = examples[(examples[name] == v)]
        r += prob_attribute_value(attribute_index, v, examples) * entropy(len(examples.columns) - 2, ex_attr_v)
    return r


def importance(attributes, examples):
    gain_vector = []
    for j in attributes:
        col_index = examples.columns.get_loc(j)
        gain = entropy(len(examples.columns.values) - 2, examples) - remainder(col_index, examples)
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
        return random.choice(max)


def get_attributes_list(examples):
    list = []
    for i in examples.columns.values:
        if (i != "class") and (i != "weight"):
            list.append(i)
    return list


def decision_tree_learning(examples, attrib, pater_examples):
    if examples.empty:
        leaf = Node(None, 'class')
        leaf.type = plurality_value(pater_examples)
        return leaf

    result = attribute_values(len(examples.columns.values) - 2, examples)
    if len(result) == 1:
        leaf = Node(None, 'class')
        leaf.type = result[0]
        return leaf

    if attrib is None:
        leaf = Node(None, 'class')
        leaf.type = plurality_value(examples)
        return leaf

    attribute = importance(attrib, examples)
    tree = Node(examples, attribute)
    col_index = examples.columns.get_loc(attribute)
    values = attribute_values(col_index, data)

    for v in values:
        exs = examples[examples[attribute] == v]
        new_attributes = copy.copy(attrib)
        new_attributes.remove(attribute)
        tree.arcs.append(v)
        tree.subtree.append(decision_tree_learning(exs, new_attributes, examples))
    return tree


# TODO quando l'albero è creato con pochi esempi, non tutti i rami si creano, e dunque certi valori non vengono trovati in arcs
def classifier(tree, example):
    while tree.current_attr != 'class':
        attribute = tree.current_attr
        value = example[attribute].values[0]
        index = tree.arcs.index(value)
        tree = tree.subtree[index]
    print('the class is ' + tree.type)


# here starts the main


training_set = data.sample(10)

attributes = get_attributes_list(training_set)

dtree = decision_tree_learning(training_set, attributes, None)

for i in range(200):

    sample = data.sample(1)

    print('\n')
    print(sample)

    classifier(dtree, sample)



