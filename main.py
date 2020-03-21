import copy
import pandas as pd
import math
import numpy


def create_data_set(file):
    data = pd.read_csv(file)
    col = len(data.columns)
    for i in range(col):
        data.columns.values[i] = 'attr' + str(i)
    data.columns.values[col - 1] = 'class'
    weight = []
    for i in range(len(data.index)):
        weight.append(1)
    data['Weight'] = weight
    return data


# elenca i possibili valori di un attributo
def attribute_values(attribute_index, examples):
    values = []
    for i in range(len(examples.index)):
        index = examples.columns.get_loc(attribute_index)
        if values.__contains__(examples.iloc[i, index]) is False:
            values.append(examples.iloc[i, index])
    return values


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
        ex_attr_v = examples[(examples[attribute_index] == v)]
        r += prob_attribute_value(attribute_index, v, examples) * entropy("class", ex_attr_v)
    return r


def importance(examples):
    gain_vector = []
    for i in examples.columns:
        gain = entropy("class", examples) - remainder(i, examples)
        gain_vector.append(copy.copy(gain))
    find_max = numpy.array(copy.copy(gain_vector))
    result = numpy.where(numpy.amax(find_max))
    return result


class Node:
    def __init__(self, examples):
        self.examples = examples
        self.subtree = []


def decision_tree_learning(examples, attributes, pater_examples):
    if examples.empty():
        return "vuoto!"  # TODO funzione plurality_value()
    result = attribute_values("class", examples)
    if len(result) == 1:
        return result[0]
    if len(attributes) == 0:
        return "rivuoto!"  # TODO anche qui plurality_value()
    attribute = examples.columns[importance(examples)]
    T = Node(examples)
    values = attribute_values(attribute, examples)
    for v in values:
        exs = examples[examples[attribute] == v]
        T.subtree.append(decision_tree_learning(exs, attributes.remove(attribute), examples))
    return T



# math.log(x,2)
# here starts the main


data = create_data_set("car.data")
print(importance(data))

