import numpy as np
import matplotlib.pyplot as plt
from utils import *
import math, copy

x_train, y_train = load_data()


def compute_cost(x, y, w, b): 
    m = x.shape[0] 
    
    total_cost = 0
    for i in range(m):
        f_wb = (x[i] * w) + b
        total_cost = total_cost + (f_wb - y[i])**2
    total_cost = (1 /(2 *m) * total_cost)
    print(total_cost)
    return total_cost



initial_w = 0
initial_b = 0

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')


def compute_gradient(x, y, w, b): 
    # Number of training examples
    m = x.shape[0]
    
    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = (x[i] * w) + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m        

    return dj_dw, dj_db

initial_w = 0
initial_b = 0

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

#compute_gradient_test(compute_gradient)