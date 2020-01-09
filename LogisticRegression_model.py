import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import math
from typing import List
import numpy as np

st.title('Logistic Regression')

st.subheader('Sigmoid Activation')
st.write('The function maps any real value into another value between 0 and 1. In machine learning, we use sigmoid to map predictions to probabilities.')
with st.echo():
    def _sigmoid(z):
        return 1.0 / (1 + np.exp(-z))


xs = np.arange(-10., 10., 0.2 )
ys: List[float] = []
for x in xs:
    ys.append(sigmoid(x))


plt.plot(xs, ys, color='green')
plt.title('Sigmoid Function')
plt.ylabel('f(x)')
plt.xlabel('x')
st.pyplot()

st.subheader('Predictions')

with st.echo():
    def predict(features, weights):
        z = np.dot(features, weights)
        return sigmoid(z)

st.subheader('Cost Function: Cross-Entropy')
with st.echo():
    def cost_function(features, labels, weights):
        observations = len(labels)
        predictions = predict(features, weights)

        class1_cost = -labels*np.log(predictions)
        class2_cost = (1-labels)*np.log(1-predictions)

        cost = class1_cost - class2_cost
        cost = cost.sum() / observations
        return cost

class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape()
        self.weights = np.zeros(n_features)
        self.bias = 0
        # gradient descent
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted-y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -=self.learning_rate * dw
            self.bias -= self.learning_rate * db


    def predict(self, X):
        linear_model = np.dot(X, weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_classes = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_classes

    def _sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))
