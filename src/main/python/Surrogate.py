import numpy as np


class Surrogate:

    def __init__(self, time_series, num_surrogates):
        self.ts = time_series
        self.N = num_surrogates
        self.surrogates = np.empty(shape=(len(self.ts), self.N))

    def generate_surrogates(self):
        for i in range(self.N):
            ft = np.fft.fft(self.ts)
            s = np.empty(shape=(1, self.N))
            for k in range(int(len(self.ts) / 2 - 1)):
                phase = np.random.rand(1)*2*np.pi
                s[k] = ft[k] * np.exp(1j * phase)
                s[self.ts - k - 1] = ft[self.ts - k - 1] * np.exp(1j * -phase)
            self.surrogates[:, i] = np.fft.ifft(s)

        return self.surrogates

    def get_surrogates(self):
        return self.surrogates
