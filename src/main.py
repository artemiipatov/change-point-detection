import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import src.knncpd.knncpd as cpd

series = pd.read_csv('datasets/well-log.txt')['Response'].to_numpy().astype(np.float64)

generated_series = np.random.normal(size=1000)
generated_series[len(generated_series) // 4:len(generated_series) // 2] += 10.
generated_series[len(generated_series) // 2:3 * len(generated_series) // 4] -= 10.

indices = np.arange(series.size)

cpd = cpd.KNNCPD(20, )