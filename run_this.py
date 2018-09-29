import csv

import numpy as np

from SALib.analyze import sobol
from SALib.sample import saltelli
from SALib.test_functions import Ishigami


def transform_csv_to_npy():
    csv_reader_input = csv.reader(open("data/inputnew.csv"))
    input_new = []
    for row in csv_reader_input:
        input_new.append(row)

    np.save("data/input_new.npy", input_new)

    csv_reader_result = csv.reader(open("data/result.csv"))
    result = []
    for row in csv_reader_result:
        result.append(row)

    np.save("data/result.npy", result)


def deal_npy():
    input_new = np.load("data/input_new.npy")
    result = np.load("data/result.npy")

    dealed_input_new = np.delete(input_new, 0, axis=0)
    dealed_input_new = np.delete(dealed_input_new, 0, axis=1)
    print(dealed_input_new[0])
    np.save("data/dealed_input_new.npy", dealed_input_new)
    dealed_result = np.delete(result, 0, axis=0)
    dealed_result = np.delete(dealed_result, 0, axis=1)
    dealed_result = np.delete(dealed_result, 0, axis=1)
    dealed_result = np.delete(dealed_result, 2, axis=1)

    np.save("data/dealed_result.npy", dealed_result)

    print(dealed_result[0])


def deal_data():
    dealed_input_new = np.load("data/dealed_input_new.npy").astype(float)

    dealed_result = np.load("data/dealed_result.npy").astype(float)
    dealed_input_new_max = np.amax(dealed_input_new, axis=0)
    dealed_input_new_min = np.amin(dealed_input_new, axis=0)

    dealed_result_max = np.amax(dealed_result, axis=0)
    dealed_result_min = np.amin(dealed_result, axis=0)
    dealed_input_new_bounds = []
    for i in range(len(dealed_input_new_min)):
        dealed_input_new_bounds.append(
            [dealed_input_new_min[i], dealed_input_new_max[i]])

    dealed_input_new_bounds = np.array(dealed_input_new_bounds)
    dealed_result_bounds = []
    for i in range(len(dealed_result_max)):
        dealed_result_bounds.append(
            [dealed_result_min[i], dealed_result_max[i]])

    dealed_result_bounds = np.array(dealed_result_bounds)

    return [dealed_result_bounds.tolist(), dealed_input_new_bounds.tolist()]


def SALib(result_title, input_title, bounds):
    num_vars = len(bounds[1]) + 1
    for i in range(len(result_title)):
        input_title.insert(0, result_title[i])
        print(input_title)
        new_bounds = bounds[1]
        new_bounds.insert(0, bounds[0][i])

        problem = {
            'num_vars': num_vars,
            'names': input_title,
            'bounds': new_bounds
        }

        # Generate samples
        param_values = saltelli.sample(problem, 1000)

        # Run model (example)
        Y = Ishigami.evaluate(param_values)

        # Perform analysis
        _ = sobol.analyze(problem, Y, print_to_console=True)

        del(input_title[0])
        del(new_bounds[0])


if __name__ == "__main__":
    # read csv and transform to npy
    # transform_csv_to_npy()

    # deal bounds
    bounds = deal_data()

    # get title
    result_title = np.delete(np.load("data/result.npy")[0][1:], 3, axis=0)[1:].tolist()
    input_title = np.load("data/input_new.npy")[0][1:].tolist()

    # sensitivity analysis 
    SALib(result_title, input_title, bounds)
