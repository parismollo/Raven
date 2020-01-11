import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

st.title('Logistic Regression')

st.subheader('Understanding Logistic Regression :books:')
st.write('Logistic regression is an algorithm used to classify observations to a discrete number of classes, 1 and 0, for example.')

st.subheader('The Idea :bulb:')
st.write('The goal is to predict whether the cancer is **benign** or **malignant** based on 10 real-valued features computed for each **cell nucleus**. To map predicted values to probabilities, we use the **sigmoid function**. The function maps any real value into another value between 0 (malignant)  and 1 (benign).')

st.subheader('The process')
st.write('To be able to predict it, we will use a linear regression like this one: ')
st.latex(r''' y = bias + W_0x_1 + W_1x_2 + ...''')
st.write('This time however we will transform the output using the sigmoid function to return a probability value between 0 and 1.')
st.latex(r''' S(y) = \frac {1} {(1 + e^{-y})}''')

st.subheader('Linear Vs Logistic')
st.write('Linear regression predictions **are continuous**, they will be used to predict  the score of a student, or the heigth of a person. In the other hand, Logistic regression predictions are discrete, they will assign a probability score related to some **discrete classification**, such as, pass or fail the exam.')

st.subheader('Optimizing our results :dart:')
st.write(''' Now to find the best predictions we need to find the values of our parameters, the weights, and bias, which **will minimize the errors of our predictions**. To measure the errors of our predictions we need a **cost function**, in linear regression, we use the sum of squared errors and then we apply the gradient descent technique to look for the best set of parameters that will give us the smallest error. However, in this scenario, the SSR cost function doesn't work.  So we will use the **cross-entropy cost** function and apply the gradient descent.

We want to optimize the cost function with respect to our parameters, weight, and bias.

We will start somewhere, compute the derivative, update the parameters, **move towards the direction until we find the cost minimum**''')

class LogisticRegression:
    def __init__(self, learning_rate=0.001, number_of_iterations=1000):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None

    # fit method
    # following Sklearn library convention
    def fit(self, X, y):
        # X is a numpy ndvector with m samples and n features
        # Y size m

        # init the parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0 # could also set to a random value


        # Gradient Descent
        for _ in range(self.number_of_iterations):
            # 1. apply linear model
            # 2. approximate with the sigmoid function
            linear_model = np.dot(X, self.weights) + self.bias
            predicted_y = self._sigmoid(linear_model)

            # to update the weights and bias we need to compute the derivative of the cost function
            dw = (1 / n_samples) * np.dot(X.T, (predicted_y - y)) # Check why is transposed
            db = (1 / n_samples) * np.sum(predicted_y - y)

            # Updating weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    def predict(self, X):

        linear_model = np.dot(X, self.weights) + self.bias
        predicted_y = self._sigmoid(linear_model)

        # define the decision boundaries
        # larger than 0.5 class 1, else class 0
        y_classes = [1 if y > 0.5 else 0 for y in predicted_y]
        return y_classes


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

st.subheader('Sklearn Breast Cancer Dataset - Model application')
st.write('Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. **They describe characteristics of the cell nucleus present in the image**')
data_load_state = st.text('Loading data...')
data = datasets.load_breast_cancer()
data_load_state.text('Loading data... Done!')

X, y = data.data, data.target

features = data.feature_names
some_data = X[:21]
some_label = y[:21]

df = pd.DataFrame(some_data, columns=features)
df['diagnosis'] = some_label

st.write(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
st.markdown('_Iterations determine the number of times we will update our parameters and move towards another point in our curve, hopefully, towards the minimum_. **Increasing the number of iterations may rise the accuracy of our model** :smile:')
number_of_iterations = st.slider('Choose the number of iterations', 0, 2000, 10)
st.markdown('_The learning rate determines the distance in which we move between iterations._')
learning_rate = st.selectbox('Choose the learning rate', (0.001, 0.005, 0.01))

model = LogisticRegression(learning_rate=learning_rate, number_of_iterations=number_of_iterations)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
st.write("Accuracy: ", accuracy(y_test, predictions))

list_of_predictions = predictions[5:15]
list_of_actual_results = y_test[5:15]
print(list_of_predictions)
print(list_of_actual_results)
s1 = pd.Series(list_of_predictions)
s2 = pd.Series(list_of_actual_results)

labels = ['Predicted diagnosis']
df1 = pd.DataFrame(s1, columns=labels)
df1['Expected diagnosis'] = s2

st.write(df1)
st.write('**Description**:  0 - malignant; 1 - benign ')
