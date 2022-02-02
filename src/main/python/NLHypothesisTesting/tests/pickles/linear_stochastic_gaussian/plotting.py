import pickle
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf
in_file = 'LSG_SMap_n100_E4_tau1_theta10_tp1.pickle'

with open(in_file, 'rb') as file:
    statistic = pickle.load(file)

l_ts = statistic.hypothesis.lambda_ts
plt.hist(statistic.hypothesis.lambda_ns, bins=20)
plt.xlabel(r'$\rho$')
plt.ylabel('n')
plt.axvline(l_ts, color='r')
plt.text(l_ts, 1, r'$\mathregular{\rho_{ts}}$: ' + str(round(l_ts, 4)), ha='right')
plt.show(block=True)
