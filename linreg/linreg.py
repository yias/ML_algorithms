
import sys
import pathlib
import scipy as sp
from scipy import stats as sts
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

if sys.platform == 'linux':
    sys.path.append(str(pathlib.Path().absolute()) + "/../stats")
else:
    sys.path.append(str(pathlib.Path().absolute()) + "\\..\\stats")
import stats as msts


def linReg(data, label, nn=False):
    """
    Function for computing the standard deviation of the data
    Input: 
        data:       a matrix of the data (number of samples x number of features. 
                    The data should be organized in a column-wise fashion,
                    meaning that each column should corespond to a feature (or
                    dimension) and each row should correspond to a sample (or
                    observation)
        
        label:      a vector containing the labels of the data. The length of
                    the vector should be equal to the number of samples of the
                    data (number of rows) and it should be an one-to-one 
                    correspondence (the 1st row of the data should correspond
                    to labels[0], the 2nd row of the data to labels[0], etc.)
    
    Output:
        slope:      the resulted slope. an 1 x n vector, where n is 
                    the number of features
        
        intersept:  the computed interspet
    """

    if data.shape[0] != label.shape[0]:
        raise Exception("the number of samples should be equal to the number of labels") 

    mu_data = msts.average(data)
    mu_label = msts.average(label)

    if len(data.shape) < 2: 
        slope =  np.dot(data - mu_data, label - mu_label) / np.dot(data - mu_data, data - mu_data)
        intersept = mu_label - slope*mu_data
        coeffs = np.array([intersept, slope])
        return coeffs
    else:
        enh_data = np.hstack([np.ones((data.shape[0], 1)), data])
        X1 = np.linalg.inv(np.dot(enh_data.T, enh_data))
        coeffs = np.linalg.multi_dot([X1, enh_data.T, label ])
        # else:
        #     enh_data = np.hstack([np.ones((data.shape[0], 1)), data ])
        #     X1 = np.linalg.inv(np.dot(enh_data.T, enh_data))
        #     coeffs = np.linalg.multi_dot([X1, enh_data.T, label ])
        slope =  np.dot((data - mu_data).T, label - mu_label) / np.dot((data - mu_data).T, data - mu_data)
        return coeffs, slope
    
    # return slope, intersept

def linReg_predict(data, coef):
     
    if len(data.shape) < 2:
        enh_data = np.array([np.ones((data.shape[0],)), data]).T
        return np.dot(enh_data, coef)
    else:
        enh_data = np.hstack([np.ones((data.shape[0], 1)), data])
        return np.dot(enh_data, coef)



x = np.random.rand(2,10) * 100
x1 = np.random.rand(100, 2) * 100
x1_label = np.zeros((x1.shape[0], 1))
x2 = 100 + np.random.rand(100, 2) * 100
x2_label = np.ones((x2.shape[0], 1))
xx = np.vstack([x1, x2])
llabels = np.vstack([x1_label, x2_label])


# slope, intercept, r_value, p_value, std_err = sts.linregress(x[0,:], x[1,:])

# coeffs1 = linReg(x[0, :].T, x[1,:].T)

# print(slope, intercept, r_value, p_value, std_err)
# print(coeffs1)

# x_t = np.linspace(0, 100, num=100)
# y_t = slope * x_t + intercept
# y_t2 = coeffs1[0] + coeffs1[1]*x_t
# y_t3 = linReg_predict(x_t, coeffs1)



# fig, ax = plt.subplots()
# ax.scatter(x[0,:], x[1,:])
# ax.plot(x_t, y_t, label='regression line', color='r')
# ax.plot(x_t, y_t2, label='my prediction', color='g')
# ax.plot(x_t, y_t3, label='my regression line', color='y')
# ax.set_title('linear regression1')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.legend()




# slope2, intercept2, r_value2, p_value2, std_err2 = sts.linregress(xx[:, 0], xx[:, 1])
# coeffs2 = linReg(xx[:, 0], xx[:, 1])

# print('slope2: ', [intercept2, slope2])
# print('coeffs2: ', coeffs2.T)



# xx_t = np.linspace(0, 200, num=100)
# yy_t = slope2 * xx_t + intercept2
# yy_t2 = coeffs2[0] + coeffs2[1]*xx_t
# yy_t3 = linReg_predict(xx_t, coeffs2)

# fig2, ax2 = plt.subplots()
# ax2.scatter(xx[:,0], xx[:,1])
# ax2.plot(xx_t, yy_t, label='regression line', color='r')
# ax2.plot(xx_t, yy_t2, label='my prediction', color='g')
# ax2.plot(xx_t, yy_t3, label='my regression line', color='y')
# ax2.set_title('linear regression2')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax2.legend()

print('xx shape', xx.shape)
print('labels shape', llabels.shape)

b_coeff, slope2 = linReg(xx, llabels)
print('b_coef', b_coeff.T)
print('slope', slope2)

reg = LinearRegression().fit(xx, llabels)
print('sk coef: ', reg.coef_)
print('sk intersept: ', reg.intercept_)

xx_t = np.linspace(0, 200, num=200)
# xx_ty = np.linspace(0, 300, num=200)

# xx_space = np.vstack([xx_t, xx_ty]).T
d1 = np.arange(0, 200, 1)
d2 = np.arange(0, 300, 1)
aa, bb = np.meshgrid(d1, d2, sparse=True)
ss = [[i,j[0]] for i in aa[0] for j in bb]
xx_space = np.array(ss)

# print(xx_space.shape)
# print(xx_space[1])
#xx_space = 
# print(xx_space)
print(type(xx_space))
print('xx_space shape: ', xx_space.shape)
space_eval = linReg_predict(xx_space, b_coeff)

print('space eval shape: ', space_eval.shape)
ttt1 = space_eval>=0.5
ttt2 = space_eval<0.5
xx_space1 = xx_space[ttt1[:,0], :]
xx_space2 = xx_space[ttt2[:,0], :]
print('xx_space1 shape ', xx_space1.shape)


yy_t = (b_coeff[1]/ b_coeff[2]) * xx_t + b_coeff[0]
# yy_t_w = (b_coeff_w[1]/ b_coeff_w[2]) * xx_t + b_coeff_w[0]

# yy_tn = (b_coeff[2]/ b_coeff[1]) * xx_t + b_coeff[0]

slope2, intercept2, rr1, rr2, rr3 = sts.linregress(xx[:, 0], xx[:, 1])
yy_tn2 =  slope2 * xx_t + intercept2
print('slope2= ', [intercept2, slope2])
coeffs2 = linReg(xx[:, 0], xx[:, 1])
print('coeffs2= ', coeffs2)
# tt0 = np.linspace(0, 200, num=100)


yy_t3 = linReg_predict(xx_t, coeffs2)
# yy_t = np.dot(np.hstack([np.ones((xx_t.shape[0], 1)), xx_t]), b_coeff)
# yy_t2 = linReg_predict(xx_t, b_coeff)


predict1 = linReg_predict(x1, b_coeff)
predict2 = linReg_predict(x2, b_coeff)
# print('predict1', predict1)
# print('predict2', predict2)
# yy_t2 = slope2[0] * xx_t


mu_1 = msts.average(x1)
mu_2 = msts.average(x2)
mus = np.vstack([mu_1, mu_2])

fig2, ax2 = plt.subplots()

ax2.scatter(xx_space1[:,0], xx_space1[:,1], color='r', alpha=0.02)
ax2.scatter(xx_space2[:,0], xx_space2[:,1], color='b', alpha=0.02)
ax2.scatter(x1[:,0], x1[:,1], label='class 1', color='b')
ax2.scatter(x2[:,0], x2[:,1], label='class 2', color='r')
ax2.plot(xx_t, yy_t, label='regression line (yy_t)', color='g')
# ax2.plot(xx_t, yy_t_w, label='yy_t_w', color='g')
ax2.plot(xx_t, yy_t3, label='yy_t3', color='y')
ax2.plot(mus[:,0], mus[:,1], label='mus line', color='k')
# ax2.plot(xx_t, yy_tn, label='yy_tn', color='g')
# ax2.plot(xx_t, yy_tn2, label='linarg yy_tn2', color='k')
ax2.set_title('classification')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()


plt.show()
