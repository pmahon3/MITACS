import matplotlib.pyplot as plt


def plot_residual_delay_map(time_series, residuals):
    plt.plot(time_series[0:len(time_series)-1], residuals[1:len(residuals)])
    plt.show()