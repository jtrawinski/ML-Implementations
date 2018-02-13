import numpy as np


class Perceptron:
    def __init__(self):
        self.data = None
        self.labels = None
        self.w = None
        self.b = None
        self.w_history = None
        self.predictedLabels = None

    def fit(self, x, y, epochs):
        """
        Binary Classification.
        Perceptron finds a separating hyperplane for the data if it exists.
        Runs until all points are correctly classified or epochs * n iterations.
        :param x: n x d numpy array of data
        :param y: length n numpy array of labels, -1 or 1
        :param epochs: number of times to loop through data
        :return: numpy array w, which determines a separating hyperplane w dot x = 0
        """
        self.data = x
        self.labels = y
        # n = number of rows
        # d = number of features
        n, d = x.shape
        # initialize w as a vector of d zeros
        w = np.repeat(0, d)
        # bias term
        b = 0
        # list of w's for each epoch
        self.w_history = [w]
        for e in range(epochs):
            misclassifications = 0
            for i in range(n):
                x_i = x[i, :]
                if y[i] * (np.dot(w, x_i) + b) <= 0:
                    # x was misclassified, update w and increase counter
                    w = w + np.dot(y[i], x_i)
                    b = b + y[i]
                    misclassifications += 1
            self.w_history.append(w)
            if misclassifications == 0:
                # all points are correctly classified
                break
        self.w = w
        self.w_history = w_history
        self.b = b
        self.data = x
        self.labels = y
        return w

    def predict(self, offset=0):
        """
        Given data and a separating hyperplane, predict -1 or 1 for the data points in x
        :param offset: optional offset for moving intercept, useful for calculating ROC curve
        :return: length n numpy array where element i is the predicted label for data point i
        """
        margin = np.dot(self.data, self.w) + self.b + offset
        pred = [1 if m > 0 else -1 for m in margin]
        self.predictedLabels = pred
        return pred

    def accuracy(self):
        """
        Determines the percentage of predictions that were correct
        :return: Percentage of correct predictions
        """
        return np.sum(self.predictedLabels == self.labels) / len(self.labels)
