import numpy as np
import math

class Neuralnetwork:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        self.weights = [(np.random.normal(0, 1, (x, y))/math.sqrt(self.sizes[0])) for x, y in zip(self.sizes[1:], self.sizes[:-1])]
        self.biases = [np.random.normal(0, 1, (x, 1)) for x in self.sizes[1:]]

        # self.weights = (np.random.normal(0, 1, (self.sizes[1], self.sizes[0]))/math.sqrt(self.sizes[0]))
        # self.biases = np.random.normal(0, 1, (self.sizes[1], 1)) 

        self.mini_batchsize = 50
        self.epochs = 200
        self.lernrate = 0.1

    def sigmoid(self, z):
        # sigmoid activation function
        return 1. / (1. + np.exp(-z))
        
    def derive_sigmoid(self, z):
        # derive the sigmoid activation functions
        return self.sigmoid(z)*(1 - self.sigmoid(z))


    def feedforward(self, a):
        # z= np.dot(self.weights, a) + self.biases
        # a = self.sigmoid(z)
        for b,w in zip(self.biases, self.weights):
            z= np.dot(w, a) + b
            a = self.sigmoid(z)
        return a


    def backpropagation(self, a, y):
        gradient_b = self.feedforward(a) - y
        gradient_w = np.dot((self.feedforward(a) - y), np.transpose(a))
        return gradient_b, gradient_w


    def update_minibatch(self, minibatch):
        gradient_biases = [np.zeros(b.shape) for b in self.biases]
        gradient_weights = [np.zeros(w.shape) for w in self.weights]

        # gradient_biases = np.zeros(self.biases.shape) 
        # gradient_weights = np.zeros(self.weights.shape) 

        for a, y in minibatch:
            gradient_b, gradient_w = self.backpropagation(a, y)
            gradient_biases += gradient_b
            gradient_weights += gradient_w
            
            gradient_biases = [gbs+gb for gbs,gb in zip(gradient_biases, gradient_b)]
            gradient_weights = [gws+gw for gws,gw in zip(gradient_weights, gradient_w)]
        
        self.biases = [b-(float(self.lernrate)/self.mini_batchsize)*grad_b
                        for b, grad_b in zip(self.biases, gradient_biases)]
        self.weights = [w-(float(self.lernrate)/self.mini_batchsize)*grad_w
                        for w, grad_w in zip(self.weights, gradient_weights)]

        # self.biases -= (float(self.lernrate)/self.mini_batchsize)*gradient_biases

        # self.weights -= (float(self.lernrate)/self.mini_batchsize)*gradient_weights



    def evaluate(self, test_data):
        n_test = len(test_data)
        res = [(np.argmax(self.feedforward(a)), np.argmax(y)) for a,y in test_data]
        return np.sum(int(i==j)/n_test for i,j in res)


    def sgd(self, training_data, test_data):
        print("nn model training...")
        # stochastic gradient descent
        n_train = len(training_data)

        for e in range(self.epochs):
            np.random.shuffle(training_data)
            np.random.shuffle(test_data)

            mini_batches = [training_data[k:k+self.mini_batchsize] for k in range(0, n_train, self.mini_batchsize)]
            self.update_minibatch(mini_batches[0])

            accurary = self.evaluate(test_data)
            # print(accurary)
            if accurary >= 0.98:
                break
        print("training completed.\nEpoch: ", e+1, "  accuracy: ", accurary*100, "%")
                

    def is_crossed(self, a):
        return np.argmax(self.feedforward(a))






            

        



