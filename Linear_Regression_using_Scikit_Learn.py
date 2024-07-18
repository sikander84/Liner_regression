import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import  load_house_data
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')


X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)


# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)

print(f"Prediction on training set sgd:\n{y_pred_sgd[:4]}" )
print(f"Target values \n{y_train[:4]}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"Prediction on training set:\n{y_pred[:4]}" )