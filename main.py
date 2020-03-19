import copy


def create_data_set(file):
    with open(file, "r") as filestream:
        data_set = []
        for line in filestream:
            currentline = line.split(",")
            data_set.append(copy.deepcopy(currentline))
    return data_set

# here starts main
