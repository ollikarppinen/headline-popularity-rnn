import util
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = util.fetch.csv()
# rng = np.append(np.arange(0,2000,100),20000)
rng = np.arange(0,10000,100)

x = df.groupby(pd.cut(df['clicks'], rng))
rows = x.count().timestamp.sum()

data = pd.DataFrame()
data[0] = rng[1:]
data[1] = list(x.count().timestamp / rows * 100)

plt.plot(data[0], data[1])
plt.title('uutisotsikoiden vierailut')
plt.xlabel('saadut vierailut')
plt.ylabel('% osuus aineistosta')
plt.yscale('log')
plt.savefig('./clicks.pdf')
