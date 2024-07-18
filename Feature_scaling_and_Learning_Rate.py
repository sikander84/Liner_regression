import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import  load_house_data, run_gradient_descent 
from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
X_train = data[:,:4]
y_train = data[:,4]

X_features = ['size(sqft)', 'bedroom', 'floors', 'age']

################ Ploting chart start #############
fig,ax = plt.subplots(1, 4, figsize=(12, 3), sharey= True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i], y_train)
    ax[i].set_xlabel(X_features[i])
    ax[i].set_ylabel("Price (1000's)")

plt.show()
################ Ploting chart End #############



_, _, hist = run_gradient_descent(X_train, y_train, 100, alpha=1e-7)




################ Ploting chart start #############
plot_cost_i_w(X_train, y_train, hist)

################ Ploting chart end #############




def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)
    print(f"mu: {mu}")
    sigma = np.std(X, axis=0)
    print(f"sigma: {sigma}")
    X_norm = (X - mu) / sigma
    return (X_norm, mu, sigma)
    
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, X_sigma = {X_sigma}")
print(f"Peak to Peak range by colum in Raw          X:{np.ptp(X_train, axis=0)}" )
print(f"Peak to Peak range by colum in Normalized   X:{np.ptp(X_norm, axis=0)}" )

w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1)


################ Ploting chart start #############

m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

    # plot predictions and targets versus original features    
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

################ Ploting chart End #############


x_house = np.array([1200, 3, 1, 40])
X_house_norm = (x_house - X_mu)/ X_sigma
print(f" house norm: {X_house_norm}")

x_house_predict = np.dot(X_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")

plt_equal_scale(X_train, X_norm, y_train)