"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np


def softmax(x):

    x = np.asarray(x)
    sh = len(x.shape)
    if sh == 1:
        x = np.row_stack(x)
        softmaxes = np.ones_like(x)
    else:
        softmaxes = np.ones_like(x)

    den = np.zeros(x.shape[1])

    for index, col in enumerate(x.T):
        for elem in col:
            den[index] = den[index] + np.exp(elem)

    for j, col in enumerate(x.T):
        for i, elem in enumerate(col):
            softmaxes[i][j] = np.exp(elem) / den[j]

    if sh == 1:
        softmaxes = np.column_stack(softmaxes)
        softmaxes = np.ravel(softmaxes)

    return softmaxes

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
