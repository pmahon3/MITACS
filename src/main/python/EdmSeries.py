import pyEDM
import pandas as pd
import matplotlib.pyplot as plt


class EdmSeries:

    def __init__(self, time_series, minE, maxE, minTp, maxTp, minTau, maxTau):
        self.ts = time_series
        self.minE = minE
        self.maxE = maxE
        self.minTp = minTp
        self.maxTp = maxTp
        self.minTau = minTau
        self.maxTau = maxTau
        self.smaps = None
        self.embeds = None

    # noinspection PyTypeChecker
    def perform_embeddings(self, variables, target, E, split, data_out_dir, drop_na, show_plot):
        # select variables
        variables = self.vars_list(variables)
        series = self.ts.iloc[:, variables]
        if drop_na:
            series.dropna(inplace=True)

        # set library and prediction to whole time series
        [lib, pred] = self.get_lib_pred(split, len(series))

        # initialize output containers
        # raw output of py.EDM.EmbedDimension
        embeds = [[0 for _ in range(self.maxTau-self.minTau)] for _ in range(self.maxTp-self.minTp)]
        # perform embedding for across tp at various tau
        for i in range(self.maxTp-self.minTp):
            for j in range(self.maxTau-self.minTau):
                embeds[i][j] = pd.DataFrame(
                    pyEDM.EmbedDimension(dataFrame=series,
                                         lib=lib,
                                         pred=pred,
                                         columns=target,
                                         target=target,
                                         maxE=E,
                                         Tp=self.minTp + i,
                                         tau=-(self.minTau + j),
                                         showPlot=show_plot)
                )
        self.embeds = embeds

        pd.DataFrame(self.embeds).to_csv(data_out_dir + "embed_plotting_data.csv")

    def perform_smaps(self, variables, target, tau, split, data_out_dir, drop_na, show_plot):
        variables = self.vars_list(variables)
        series = self.ts.iloc[:, variables]
        if drop_na:
            series.dropna(inplace=True)

        [lib, pred] = self.get_lib_pred(split, len(series))
        self.smaps = [[0 for _ in range(self.maxTp-self.minTp)] for _ in range(self.maxE-self.minE)]

        for i in range(self.maxE-self.minE):
            print("E: " + str(i + 1))
            for j in range(self.maxTp-self.minE):
                self.smaps[i - 1][j - 1] = pd.DataFrame(
                    pyEDM.PredictNonlinear(
                        dataFrame=series,
                        lib=lib,
                        pred=pred,
                        E=self.minE + i,
                        Tp=self.minTp + j,
                        tau=tau,
                        columns=target,
                        target=target,
                        showPlot=show_plot)
                )

        pd.DataFrame(self.smaps).to_csv(data_out_dir + "data.csv")

    def embed_plotting(self, plot_out_dir):
        embeds = self.embeds

        # averaged output for rho vs. tp for a given E
        tp_curves = [[0 for _ in range(self.maxTp-self.minTp)] for _ in range(self.maxE-self.minE)]
        # averaged output if rho vs tp
        avg = [0 for _ in range(self.maxTp-self.minTp)]

        # plot rho vs. E for embeddings (varied over tau) for a given tp
        for i in range(self.maxTp-self.minTp):
            print(i)
            avg_ = embeds[i][1]
            for j in range(self.maxTau-self.minTau):
                if j == 1:
                    continue
                else:
                    avg_['rho'] = avg_['rho'] + embeds[i][j]['rho']
                plt.plot(embeds[i][j]['E'], embeds[i][j]['rho'], color='blue',
                         alpha=(self.maxTau-self.minTau - j) / (self.maxTau-self.minTau))
                plt.xlabel('E')
                plt.ylabel('rho')
            avg_['rho'] = avg_['rho'] / 10
            avg[i] = avg_
            plt.plot(avg_['E'], avg_['rho'], color='red', alpha=1)
            plt.title("Prediction Skill for Various Embeddings (Tau) with Tp = " + str(i + 1))
            plt.savefig(plot_out_dir + "tp_" + str(i + 1) + ".png")
            plt.cla()

        # plot rho vs. E for means of embeddings for a given tp
        for i in range(self.maxTp-self.minTp):
            plt.plot(avg[i]['E'], avg[i]['rho'], color='red', alpha=(self.maxTp-self.minTp - i) / (self.maxTp-self.minTp))
        plt.title("Mean Prediction Skill for Varying Tp")
        plt.xlabel('E')
        plt.ylabel(r"$\rho$")
        plt.savefig(plot_out_dir + "average_E_skill" + "_2020.png")
        plt.cla()

        # compute rho vs. E across various embeddings (tau)
        for i in range(self.maxE-self.minE):
            for j in range(self.maxTp-self.minE):
                tp_curves[i][j] = avg[j].iloc[i]['rho']
            plt.plot(avg[1]['E'], tp_curves[i], color='blue', alpha=(self.maxE-self.minE - i) / (self.maxE-self.minE))
        plt.xlabel('Tp')
        plt.ylabel(r'$\rho$')
        plt.title('Time to Prediction Skill for Various Embeddings (E)')
        plt.savefig(plot_out_dir + "average_tp_skill.png")

    def smaps_plotting(self,  plot_out_dir):
        avg = [0 for _ in range(self.maxE-self.minE)]
        for i in range(self.maxE-self.minE):
            avg_ = self.smaps[i][1]
            for j in range(self.maxTp-self.minTp):
                if j == 1:
                    continue
                else:
                    avg_['rho'] = avg_['rho'] + self.smaps[i][j]['rho']
                plt.plot(self.smaps[i][j]['Theta'], self.smaps[i][j]['rho'], color='blue',
                         alpha=(self.maxTp-self.minTp - j) / (self.maxTp-self.minTp))
                plt.ylabel(r'\rho')
            avg_['rho'] = avg_['rho'] / 10
            avg[i] = avg_
            plt.plot(avg_['Theta'], avg_['rho'], color='red', alpha=1.0)
            plt.title("E = " + str(i + 1))
            plt.savefig(plot_out_dir + "E_" + str(i + 1) + "_2020.png")
            plt.cla()

        for i in range(self.maxE-self.minE):
            plt.plot(avg[i]['Theta'], avg[i]['rho'], color='red', alpha=(self.maxE-self.minE - i) / (self.maxE-self.minE))
        plt.title("Mean Prediction Skill for Varying E")
        plt.xlabel(r'$\theta$')
        plt.ylabel(r"$\rho$")
        plt.savefig(plot_out_dir + "average" + "_2020.png")
        plt.cla()

    def vars_list(self, variables):
        out = []
        for var in variables:
            out.append(self.ts.columns.get_loc(var))
        return out

    @staticmethod
    def get_lib_pred(split, length):
        spl = round(split * round(length))
        lib = "1 " + str(spl)
        pred = str(spl + 1) + " " + str(length)
        return [lib, pred]
