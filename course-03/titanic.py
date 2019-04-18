
import pandas as pd
import numpy as np
import csv
import os
import math
import random

data_path = "/Users/heany/code/AI-For-NLP/dataset/titanic/train.csv"
# print(os.path.exists(data_path))

data = pd.read_csv(data_path)
data = data.dropna()

index = (data["Age"] > 20) & (data["Fare"] > 130) & (data["Fare"]<400)

age = data[index]["Age"]
fare = data[index]["Fare"]

min_error_rate = float('inf')

loop_times = 10000

def func(age, k, b):
    return k * age + b

def loss(y, yhat):
    return np.mean(np.abs(y - yhat))
    # return np.mean(np.square(y - yhat))
    # return np.mean(np.sqrt(y - yhat))

def train_random():
    Loop_times = loop_times
    Min_error_rate = min_error_rate
    while Loop_times > 0:
        k_hat = random.random() * 20 - 10
        b_hat = random.random() * 20 - 10
        estimated_fares = func(age, k_hat, b_hat)
        error_rate = loss(y=fare, yhat=estimated_fares)

        if error_rate < Min_error_rate:
            Min_error_rate = error_rate
            best_k, best_b = k_hat, b_hat
            print('loop == {}'.format(10000-Loop_times))
                # losses.append(min_error_rate)
            print('f(age) = {} * age + {}, with error rate: {}'.format(best_k, best_b, error_rate))
        Loop_times -= 1


def step(): return random.random() * 1

def train_supervised_direction():
    change_directions = [
        # (k, b)
        (+1, -1), # k increase, b decrease
        (+1, +1),
        (-1, +1),
        (-1, -1)  # k decrease, b decrease
    ]

    k_hat = random.random() * 20 - 10
    b_hat = random.random() * 20 - 10

    Loop_times = loop_times
    Min_error_rate = min_error_rate
    direction = random.choice(change_directions)
    losses = []

    best_k, best_b = k_hat, b_hat
    best_direction = None

    while Loop_times >0:

        k_delta_direction, b_delta_direction = direction
        
        k_delta = k_delta_direction * step()
        b_delta = b_delta_direction * step()
        
        new_k = best_k + k_delta
        new_b = best_b + b_delta



        estimated_fares = func(age, new_k, new_b)
        error_rate = loss(y=fare, yhat=estimated_fares)

        if error_rate < Min_error_rate:
            Min_error_rate = error_rate
            best_k, best_b = new_k, new_b

            direction = (k_delta_direction, b_delta_direction)

            print('loop == {}'.format(10000-Loop_times))
            print('f(age) = {} * age + {}, with error rate: {}'.format(best_k, best_b, error_rate))
        else:
            direction = random.choice(list(set(change_directions) - {(k_delta_direction, b_delta_direction)}))

        losses.append(error_rate)

        Loop_times -= 1

    pass

def derivate_k(y, yhat, x):
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]

    return np.mean([a * -x_i for a, x_i in zip(abs_values, x)])


def derivate_b(y, yhat):
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]
    return np.mean([a * -1 for a in abs_values])

def train_gradient_descent():
    Loop_times = loop_times
    Min_error_rate = min_error_rate

    learing_rate = 1e-1
    losses = []

    k_hat = random.random() * 20 - 10
    b_hat = random.random() * 20 - 10

    while Loop_times > 0:

        k_delta = -1 * learing_rate * derivate_k(fare, func(age, k_hat, b_hat), age)
        b_delta = -1 * learing_rate * derivate_b(fare, func(age, k_hat, b_hat))

        k_hat += k_delta
        b_hat += b_delta

        best_k, best_b = k_hat, b_hat


        estimated_fares = func(age, k_hat, b_hat)
        error_rate = loss(y=fare, yhat=estimated_fares)


        print('loop == {}'.format(10000-Loop_times))
        print('f(age) = {} * age + {}, with error rate: {}'.format(best_k, best_b, error_rate))

        losses.append(error_rate)

        Loop_times -= 1


    pass


#+++++++++++++++++  training  ++++++++++++++++++++
# train_random()
# train_supervised_direction()
train_gradient_descent()