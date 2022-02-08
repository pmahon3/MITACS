import matplotlib.pyplot as plt
import numpy as np


def plot_residual_delay_map(data, title: str = '', x_label: str = '', y_label: str = ''):
    cs = np.arange(len(data)-1)
    plt.scatter(data[0:-1, 0], data[1:, 1], c=cs)
    plt.title(title)
    plt.xlabel(xlabel=x_label)
    plt.ylabel(ylabel=y_label)
    plt.show()
