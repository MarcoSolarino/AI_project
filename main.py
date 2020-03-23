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


# elenca i possibili valori di un attributo
def attribute_values(attribute_index, examples):
    values = []
    for i in range(len(examples.index)):
        #  index = examples.columns.get_loc(attribute_index)
        if values.__contains__(examples.iloc[i][attribute_index]) is False:
            values.append(examples.iloc[i][attribute_index])
    return values


def split_data_set(attribute_index, examples):
    df_tuple = dict(tuple(examples.index.groupby(attribute_index)))
    return df_tuple


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
    def __init__(self, examples):
        self.examples = examples
        self.subtree = []


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
    print(attrib)
    if examples.empty:
        return plurality_value(pater_examples)
    result = attribute_values(len(examples.columns.values) - 2, examples)
    if len(result) == 1:
        return result[0]
    if attrib is None:
        return plurality_value(examples)
    attribute = importance(attrib, examples)
    tree = Node(examples)
    col_index = examples.columns.get_loc(attribute)
    values = attribute_values(col_index, examples)
    for v in values:
        exs = examples[examples[attribute] == v]
        new_attributes = copy.copy(attrib)
        new_attributes.remove(attribute)
        tree.subtree.append(decision_tree_learning(exs, new_attributes, examples))
    return tree


# here starts the main


data = create_data_set("car.data")

attributes = get_attributes_list(data)

vettore_prova = ['attr0', 'attr1', 'attr2', 'attr4']

# print(importance(vettore_prova, data))

decision_tree_learning(data, attributes, None)



