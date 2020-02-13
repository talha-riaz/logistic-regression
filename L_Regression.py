import numpy as np

class LogisticRegression:

    def __init__(self):
        self.w = []
    # method to train the logistic regression model
    def fit(self, X, y, l=0.01, learningRate=0.01, eps=0.01):
        X = np.concatenate((np.array([[1.]]*X.shape[0]),X),axis=1)
        return self.__batchGradDescent(X, y, l, learningRate, eps)
    # given a set of features, returns the prediction based on the trained model
    def predict(self, X):
        X = np.concatenate((np.array([[1.]]*X.shape[0]),X),axis=1)
        return np.array([int(yH > 0.5) for yH in self.__sigmoid(self.w.transpose().dot(X.transpose()))])
    # given the labels and the predicted labels, returns the accuracy of the model
    def evaluate_acc(self, y, yH):
        return np.mean([int(y[i] == yH[i]) for i in range(len(y))])
    # implementation of the batch gradient descent
    def __batchGradDescent(self, X, y, l, learningRate, eps):
        t = 0
        N,D = X.shape
        self.w = np.zeros(D)
        g = np.inf
        accuracy = []
        while np.linalg.norm(g) > eps:
            if t % 100 == 0:
                accuracy.append(self.evaluate_acc(y, self.predict(X[:,1:])))
            g = self.__grad(X, y, self.w, l)
            self.w = self.w - g*learningRate
            t += 1
        return accuracy
    # returns the gradient, used by __batchGradDescent
    def __grad(self, X, y, w, l):
        return X.transpose().dot(self.__sigmoid(w.transpose().dot(X.transpose())) - y)/X.shape[0] + l*w
    #the logistic function for logistic regression
    def __sigmoid(self, x):
        return 1./(1. + np.exp(-x))
