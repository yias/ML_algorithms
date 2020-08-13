

import sys
import numpy as np
import matplotlib.pyplot as plt
# from stats import stats as st
from scipy.stats import norm
from scipy.stats import skewnorm
from scipy import stats
from distributions import gaussDistribution



data1 = np.random.randint(low=0, high=20, size=5) / 10.0

data2 = np.random.randint(low=20, high=40, size=15) / 10.0

data3 = np.random.randint(low=40, high=70, size=10) / 10.0

data4 = np.random.randint(low=70, high=100, size=5) / 10.0

data = np.hstack([data1, data2, data3, data4])

print(data.shape)

eval_data = np.linspace(-5, 5, num=200)
# eval_data = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
# print(eval_data)

gD = gaussDistribution.generalNormDist(0)
print(gD.pdf(0))
gD1 = gaussDistribution.generalNormDist(loc=0, skew=3)
gD2 = gaussDistribution.generalNormDist(loc=0, skew=-3)
gD3 = gaussDistribution.generalNormDist(loc=0, skew=-.3)
gD4 = gaussDistribution.generalNormDist(loc=0, skew=.3)
gD5 = gaussDistribution.generalNormDist(loc=0, skew=6)

gD_o1 = gaussDistribution.generalNormDist(loc=1, omega=.3)
gD_o2 = gaussDistribution.generalNormDist(loc=0, omega=2)
gD_o3 = gaussDistribution.generalNormDist(loc=0, omega=5)

tt = gD.pdf(eval_data)

gD_t = gaussDistribution.generalNormDist(loc=0, skew=3)


num_bins = 10
fig, ax = plt.subplots()
n, bins, patches = ax.hist(data, num_bins, facecolor='blue', alpha=0.5, label='data')

e_alpha, e_loc, e_omega = stats.skewnorm.fit(data)
t_eval = np.linspace(np.min(data) - 2, np.max(data))
tt_s = stats.skewnorm(e_alpha, e_loc, e_omega)
ax.plot(t_eval, tt_s.pdf(t_eval)*10, label='scipy')

gD_t = gaussDistribution.generalNormDist(loc=e_loc, skew=e_alpha, omega=e_omega)
ax.plot(t_eval, gD_t.pdf(t_eval)*10, label='mine')

ax.legend()

####################
fig, ax5 = plt.subplots()
tt_s2 = stats.skewnorm(0.1, 2, 1)
gD_t2 = gaussDistribution.generalNormDist(loc=2, skew=0.1, omega=1)
t_eval2 = np.linspace(0, 4)
ax5.plot(t_eval2, gD_t2.pdf(t_eval2), label='mine')
ax5.plot(t_eval2, tt_s2.pdf(t_eval2), label='scipy')
ax5.legend()
####################

fig, ax6 = plt.subplots()
# tt_s2 = stats.skewnorm(0.1, 2, 1)
# gD_t2 = distributions.gaussDistribution.generalNormDist(loc=2, skew=0.1, omega=1)
# t_eval2 = np.linspace(0, 4)
ax6.plot(t_eval2, gD_t2.loglkd(t_eval2), label='mine')
ax6.plot(t_eval2, tt_s2.logpdf(t_eval2), label='scipy')
ax6.set_title('log-likelihood')
ax6.legend()
####################

fig, ax2 = plt.subplots(3)
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

ax2[2].plot(eval_data, gD.pdf(eval_data), color='b', label='omega=1')
ax2[2].plot(eval_data, gD_o1.pdf(eval_data), color='r', label='omega=0.3')
ax2[2].plot(eval_data, gD_o2.pdf(eval_data), color='g', label='omega=2')
ax2[2].plot(eval_data, gD_o3.pdf(eval_data), color='m', label='omega=5')
ax2[2].legend()

# print(gD.norm_cdf(eval_data))

fig, ax3 = plt.subplots(2)
ax3[0].plot(eval_data, gD.norm_cdf(eval_data), label='my cdf')
ax3[0].plot(eval_data, norm.cdf(eval_data), label='scipy cdf')
ax3[0].plot(eval_data, gD.norm_cdf(-0.3*eval_data), label='my cdf(alpha=-0.3)')
ax3[0].plot(eval_data, norm.cdf(-0.3*eval_data), label='scipy cdf(alpha=-0.3)')
ax3[0].legend()
ax3[0].set_title('cdf')

ax3[1].plot(eval_data, norm.pdf(eval_data))
ax3[1].plot(eval_data, gD.pdf(eval_data))
ax3[1].plot(eval_data, norm.pdf(-0.3*eval_data))
ax3[1].plot(eval_data, gD.pdf(-0.3*eval_data))

ax3[1].set_title('pdf')

#####################################

data2d = np.array([data, data]).T

e_data2d = np.array([eval_data, eval_data])
gD = gaussDistribution.generalNormDist(loc=[0,0])
# print(st.covat(e_data2d))

pdf_t =gD.norm_pdf(e_data2d.T)

# print(pdf_t.shape)
# print(pdf_t)

fig, ax7 = plt.subplots()
ax7.plot(e_data2d[0, :], pdf_t)
# ax7[1].plot(e_data2d[1, :], pdf_t[:,1])


plt.show()