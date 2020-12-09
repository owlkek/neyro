import math
import random
import pandas
import numpy as np
import matplotlib.pyplot as plt
# скачать анакоду, запустить Юпитер, новый файл, всатвить код и запуск на shift + enter
class Neuron:
    def __init__(self):
        self.ro = 0.0005
        self.epochs = 5000
        self.eps = 1
        self.weight = 0
        self.input = 0
        self.bias = 0

    def set_random_weights(self, mn, mx):
        self.weight = random.uniform(mn, mx)
        self.bias = random.uniform(mn, mx)

    def set_input(self, x):
        self.input = x

    def forward(self):
        return self.bias + self.input * self.weight

    def correct(self, delta):
        self.weight += delta * self.ro * self.input
        self.bias += self.ro * delta

    def train(self, X, Y):
        can = True
        for i in range(0, self.epochs):
            can = True
            for j in range(0, len(X)):
                self.set_input(X[j])
                result = self.forward()
                expected = Y[j]
                if abs(expected - result) > self.eps:
                    can = False
                    self.correct(expected - result)
            if can:
                break



data = pandas.read_csv('data.csv')
X = list(data.iloc[:,0])
Y = list(data.iloc[:,1])

# Метод наименьших квадратов
X_mean = np.mean(X)
Y_mean = np.mean(Y)

num = 0
den = 0
for i in range(len(X)):
    num += (X[i] - X_mean)*(Y[i] - Y_mean)
    den += (X[i] - X_mean)**2
k = num / den
b = Y_mean - k*X_mean

# График наименьших квадратов
plt.plot([min(X), max(X)], [min(X)*k + b, max(X)*k + b], color='green')

# Нейрон
n = Neuron()
n.set_random_weights(-1.0, 1.0)
n.train(X, Y)

k_n = n.weight
b_n = n.bias

# График нейросети
plt.rcParams['figure.figsize'] = (12.0, 9.0)
plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(X)*k_n + b_n, max(X)*k_n + b_n], color='red')
plt.show()
