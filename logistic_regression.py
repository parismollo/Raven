import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

st.sidebar.title('Logistic Regression')

st.subheader('Understanding Logistic Regression')
st.write('Logistic regression is an algorithm used to classify observations to a discrete number of classes, 1 and 0, for example.')

st.subheader('Linear Vs Logistic')
st.write('Linear regression predictions **are continuous**, they will be used to predict  the score of a student, or the heigth of a person. In the other hand, Logistic regression predictions are discrete, they will assign a probability score related to some **discrete classification**, such as, pass or fail the exam.')

# st.subheader('Building a Logistic Regression model')

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

data_load_state = st.text('Loading data...')
data = datasets.load_breast_cancer()
data_load_state.text('Loading data... Done!')

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

number_of_iterations = st.slider('Choose the number of iterations', 0, 2000, 10)
learning_rate = st.selectbox('Choose the learning rate', (0.001, 0.005, 0.01))

model = LogisticRegression(learning_rate=learning_rate, number_of_iterations=number_of_iterations)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
st.write("Accuracy: ", accuracy(y_test, predictions))
