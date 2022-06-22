import numpy as np
import csv
from sympy import *

headers = ["x0", "x1", "x2", "x3", "y"]

n_samples = 1000

data_x = np.zeros((n_samples, 4))
data_y = np.zeros((n_samples, 1))


def protected_division(x, y):
    return float(float(np.sign(y))*(x / (abs(y) + 10e-6)))

targets = []

with open("data/datasets/fake_data.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=",")
    csvwriter.writerow([n_samples, 4, '', '', ''])
    csvwriter.writerow(headers)
    for i in range(n_samples):
        x0 = np.random.randn()*0.25 + 0
        x1 = np.random.randn()*0.25 + 1
        x2 = np.random.randn()*0.25 + 2
        x3 = np.random.randn()*0.25 + 3

        data_x[i] = np.array([x0, x1, x2, x3])

        data_y[i] = ((x0**4) - (x1**4)) + ((x2**4) / (x3**4))
        targets.append(data_y[i][0])

        csvwriter.writerow([x0, x1, x2, x3, data_y[i][0]])

print("Variance: ", np.array(targets).var())

inps = ["input_{}".format(x) for x in range(4)]
variables = symbols(inps)
print(simplify("((x0**4) - (x1**4)) + ((x2**4) / (x3**4))"))