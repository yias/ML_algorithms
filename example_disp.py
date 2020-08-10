

import sys
import numpy as np
import matplotlib.pyplot as plt
import distributions.gaussDistribution


data1 = np.random.randint(low=0, high=20, size=5) / 10.0

data2 = np.random.randint(low=20, high=40, size=15) / 10.0

data3 = np.random.randint(low=40, high=70, size=10) / 10.0

data4 = np.random.randint(low=70, high=100, size=5) / 10.0

data = np.hstack([data1, data2, data3, data4])

print(data.shape)

eval_data = np.linspace(0, 10)
# eval_data = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
print(eval_data)

gD = distributions.gaussDistribution.gaussDistribution(5.0, 2.0, skew=5)

tt = gD.pdf(eval_data)

num_bins = 10
fig, ax = plt.subplots()
n, bins, patches = ax.hist(data, num_bins, facecolor='blue', alpha=0.5)
print(n)
print(np.max(n))
# print(tt)
ax.plot(eval_data, tt*10)

fig, ax2 = plt.subplots()
ax2.plot(gD.cdf(eval_data))
plt.show()