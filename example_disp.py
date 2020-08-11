

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import distributions.gaussDistribution


data1 = np.random.randint(low=0, high=20, size=5) / 10.0

data2 = np.random.randint(low=20, high=40, size=15) / 10.0

data3 = np.random.randint(low=40, high=70, size=10) / 10.0

data4 = np.random.randint(low=70, high=100, size=5) / 10.0

data = np.hstack([data1, data2, data3, data4])

print(data.shape)

eval_data = np.linspace(-5, 5, num=200)
# eval_data = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
# print(eval_data)

gD = distributions.gaussDistribution.gaussDistribution(0)
print(gD.pdf(0))
gD1 = distributions.gaussDistribution.gaussDistribution(mu=0, skew=3)
gD2 = distributions.gaussDistribution.gaussDistribution(mu=0, skew=-3)
gD3 = distributions.gaussDistribution.gaussDistribution(mu=0, skew=-.3)
gD4 = distributions.gaussDistribution.gaussDistribution(mu=0, skew=.3)
gD5 = distributions.gaussDistribution.gaussDistribution(mu=0, skew=6)

gD_o1 = distributions.gaussDistribution.gaussDistribution(mu=1, omega=.3)
gD_o2 = distributions.gaussDistribution.gaussDistribution(mu=0, omega=2)
gD_o3 = distributions.gaussDistribution.gaussDistribution(mu=0, omega=5)

tt = gD.pdf(eval_data)

num_bins = 10
fig, ax = plt.subplots()
n, bins, patches = ax.hist(data, num_bins, facecolor='blue', alpha=0.5)
print(n)
print(np.max(n))
# print(tt)
ax.plot(eval_data, tt*10)

fig, ax2 = plt.subplots(2)
ax2[0].plot(eval_data, gD2.cdf(eval_data), color='g', label='alpha=-3')
ax2[0].plot(eval_data, gD3.cdf(eval_data), color='m', label='alpha=-0.3')
ax2[0].plot(eval_data, gD.cdf(eval_data), color='b', label='alpha=0')
ax2[0].plot(eval_data, gD4.cdf(eval_data), color='y', label='alpha=0.3')
ax2[0].plot(eval_data, gD1.cdf(eval_data), color='r', label='alpha=3')
ax2[0].legend()

ax2[1].plot(eval_data, gD2.pdf(eval_data), color='g', label='alpha=-3')
ax2[1].plot(eval_data, gD3.pdf(eval_data), color='m', label='alpha=-0.3')
ax2[1].plot(eval_data, gD.pdf(eval_data), color='b', label='alpha=0')
ax2[1].plot(eval_data, gD4.pdf(eval_data), color='y', label='alpha=0.3')
ax2[1].plot(eval_data, gD1.pdf(eval_data), color='r', label='alpha=3')
ax2[1].plot(eval_data, gD5.pdf(eval_data), color='c', label='alpha=5')
ax2[1].legend()

# ax2[2].plot(eval_data, gD.pdf(eval_data), color='b', label='omega=1')
# ax2[2].plot(eval_data, gD_o1.pdf(eval_data), color='r', label='omega=0.3')
# ax2[2].plot(eval_data, gD_o2.pdf(eval_data), color='g', label='omega=2')
# ax2[2].plot(eval_data, gD_o3.pdf(eval_data), color='m', label='omega=5')
# ax2[2].legend()

# print(gD.norm_cdf(eval_data))

fig, ax3 = plt.subplots(2)
ax3[0].plot(eval_data[1:], gD.norm_cdf(eval_data), label='my cdf')
ax3[0].plot(eval_data, norm.cdf(eval_data), label='scipy cdf')
ax3[0].plot(eval_data[1:], gD.norm_cdf(-0.3*eval_data), label='my cdf(alpha=-0.3)')
ax3[0].plot(eval_data, norm.cdf(-0.3*eval_data), label='scipy cdf(alpha=-0.3)')
ax3[0].legend()
ax3[0].set_title('cdf')

ax3[1].plot(eval_data, norm.pdf(eval_data))
ax3[1].plot(eval_data, gD.pdf(eval_data))
ax3[1].plot(eval_data, norm.pdf(-0.3*eval_data))
ax3[1].plot(eval_data, gD.pdf(-0.3*eval_data))

ax3[1].set_title('pdf')

plt.show()