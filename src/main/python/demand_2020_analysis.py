import numpy as np
import pandas as pd
import pyEDM
import matplotlib.pyplot as plt

inFile = './main/resources/ontario_averaged_2020.csv'
outDir = './main/resources/plots/2020/'
maxTp = 10
maxTau = 10

delta_ts = pd.read_csv('./main/resources/ontario_averaged_first_diff_2020.csv')['Demand']
time = pd.DataFrame(np.linspace(2, 365, 365))
delta_ts = pd.concat([time, delta_ts[2:len(delta_ts)]], axis=1)
delta_ts.columns = ['Time', 'Demand']

embedDimensions = [[0 for i in range(maxTau)] for j in range(maxTp)]

for i in range(maxTp):
    for j in range(maxTau):
        embedDimensions[i - 1][j - 1] = pd.DataFrame(
            pyEDM.EmbedDimension(dataFrame=delta_ts,
                                 lib="2 365",
                                 pred="2 365",
                                 columns="Demand",
                                 target="Demand",
                                 maxE=10,
                                 Tp=i+1,
                                 tau=-(j+1),
                                 showPlot=False)
        )

for i in range(maxTp):
    total = embedDimensions[i][1]
    for j in range(maxTau):
        if i == 1:
            continue
        else:
            total = total + embedDimensions[i][j]
        plt.plot(embedDimensions[i][j]['rho'], color='blue', alpha=0.3)
        plt.xlabel('E')
        plt.ylabel('rho')
    avg = total / 10
    plt.plot(avg['rho'], color='red', alpha=0.7)
    plt.title("Tp = " + str(i + 1) + '\n 2020 Demand')
    plt.savefig(outDir + "/demand_embed_dim_tp_" + str(i+1) + "_2020.png")
    plt.cla()
