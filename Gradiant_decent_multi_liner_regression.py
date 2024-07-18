import numpy as np
import copy, math 
np.set_printoptions(precision=2)
X_train = np.array([[2104, 5 , 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460,232,178])



b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])


def predict(x, w, b):
    p = np.dot(x,w) + b
    return p

x_vec = X_train[0,:]


f_wb = predict(x_vec, w_init, b_init)

print (f"f_wb shape:  {f_wb.shape}, f_wb perdiction: {f_wb}")


def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost/ (2 * m)
    return cost

cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')


def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n,)
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i] , w )+ b) - y[i]
        for j in range(n):
            dj_dw[j] =  dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


temp_dj_dw, temp_dj_db = compute_gradient(X_train, y_train, w_init, b_init)


print(f'dj_db at the inital w,b : {temp_dj_db}')
print(f'dj_dw at the inital w,b : \n {temp_dj_dw}')

def gradient_decent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):

    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 10000:
            J_history.append(cost_function(X, y, w, b))
        if i% math.ceil(num_iters/10) == 0:
             print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    return w, b, J_history


initial_w = np.zeros_like(w_init)
inital_b = 0.

iterations = 1000
alpha = 5.0e-7

w_final, b_final, J_hist = gradient_decent(X_train, y_train, initial_w, inital_b, compute_cost, compute_gradient, alpha, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")


m,_ = X_train.shape

for i in range(m):
    print(f"Perdiction: {np.dot(X_train[i], w_final) + b_final:0.2f},  Expected Cost: {y_train[i]}")
   