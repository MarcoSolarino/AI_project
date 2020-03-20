import copy
import pandas as pd
import math


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
        if values.__contains__(examples.iloc[i, attribute_index]) is False:
            values.append(examples.iloc[i, attribute_index])
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
        c = examples.columns[attribute_index]
        ex_attr_v = examples[(examples[c] == v)]
        r += prob_attribute_value(attribute_index, v, examples) * entropy(len(ex_attr_v.columns) - 2, ex_attr_v)
    return r


def importance(attribute_index, examples):
    gain = entropy(len(examples.columns) - 2, examples) - remainder(attribute_index, examples)
    return gain


# math.log(x,2)
# here starts the main


data = create_data_set("car.data")
print(importance(0, data))

