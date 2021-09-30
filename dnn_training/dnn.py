import numpy as np
import math

class NeuralNetwork_training():

    '''
    Bring the training data into this class for training, and get the weight and bias.
    The training method is mini batch method.
    '''

    def __init__(self, NeuralNetwork_sizes,training_data, testing_data,target_accuracy):
        # we use a network with two layers for this problem
        self.sizes = NeuralNetwork_sizes
        self.inputs_num = NeuralNetwork_sizes[0]
        self.outputs_num = NeuralNetwork_sizes[1]

        # Initialize weight matrix and bias vector (with N (0, 1) Standard normal distribution) distributed values.
        #The random number sequence generated is fixed
        np.random.seed(0)
        # Additionally divide the weights by sqrt(N0)
        self.weights = (np.random.normal(0, 1,(self.sizes[1],self.sizes[0]))) /math.sqrt(self.sizes[0])
        self.biases =np.random.normal(0, 1, (self.sizes[1],1))
        self.parameter = (self.biases,self.weights)

        # Number of mini batch, Number of iterations and learning rate
        self.mini_batchsize = 50
        self.epochs_num_max = 30
        self.eta = 0.1

        self.training_data = training_data
        self.testing_data = testing_data

        # Number of elements per input, for training_data and testing_data is the same.
        self.input_elements_num = int(len(self.training_data[0]))
        # Total number of training samples
        self.training_num = int(len(self.training_data)//2)
        self.training_inputs = self.training_data[:self.training_num]
        self.training_outputs = self.training_data[self.training_num:]

        # Total number of testing samples
        self.testing_num = int(len(self.testing_data)//2)
        self.testing_inputs = self.testing_data[:self.testing_num]
        self.testing_outputs = self.testing_data[self.testing_num:]

        self.target_accuracy = target_accuracy

    def sigmoid(self,z):
        '''
        the activation function: f(x)= 1/(1 + e^(-x))

        '''
        return 1/(1 + np.exp(-z))

    def feedforward(self,input):
        '''
        the inputs are directly through a series of weights and biases to the outputs.
        '''
        z = np.dot(self.weights, input) + self.biases
        return self.sigmoid(z)

    def costfunction(self, input, output):
        '''
        a measure of 'how good' the neural network is
        We choose the 'Cross-Entropy-Cost'
        '''
        a1 = self.feedforward(input)
        cost = np.zeros((a1.shape[0]))

        for i in range(a1.shape[0]):
            cost[i] = -output[i]*np.log(a1[i])-(1-output[i])*np.log(1-a1[i])
        return np.sum(cost)

    def backpropagation(self,input,output) ->tuple:
        '''
        compute the gradient of the cost function with respect to the biases and weights of the network
        '''
        # Determine the gradient of cost function for biases.
        grad_b = self.feedforward(input) - output

        # Determine the gradient of cost function for weights.
        grad_w = np.dot((self.feedforward(input) - output),np.transpose(input))

        return grad_b,grad_w

    def output_attributes(self,input):
        '''
        Judge the output attribute by input, crossed or not.
        '''
        a1 = self.feedforward(input)
        attributes = np.zeros(np.shape(a1))
        attributes[np.argmax(a1)] = 1
        #The number of attribute types is 2.
        if a1[0] == a1[1]:
            attributes[1] = 1

        return attributes

    def outputs_attributes(self,bogens_inputs_list)->list:
        '''
        Judge the outputs attributes by inputs, crossed or not.
        Read the gray value of the image with multiple sub-folders. So it goes through multiple cycles.
        '''
        bogens_inputs_attribute = []
        for bogen_inputs in bogens_inputs_list:

            bogen_inputs_attribute = []
            for que_inputs in bogen_inputs:

                que_inputs_attribute = []
                for ans_input in que_inputs:
                    a1 = self.feedforward(ans_input)
                    attribute = np.zeros(np.shape(a1))
                    attribute[np.argmax(a1)] = 1
                    #The number of attribute types is 2.
                    if a1[0] == a1[1]:
                        attribute[1] = 1
                    que_inputs_attribute.append(attribute)

                bogen_inputs_attribute.append(que_inputs_attribute)

            bogens_inputs_attribute.append(bogen_inputs_attribute)

        return bogens_inputs_attribute

    def get_mini_batches(self,seed=1) ->list:
        '''
        A randomly selected k(50)-element of the training data is considered to be a mini-batch.
        Method: Rearrange the training set. With mini batch size(50) as the unit to divide the rearranged training set.
        '''

        #The random number sequence generated is fixed
        np.random.seed(seed)

        #Randomly determine the new arrangement order.
        distribution = list(range(self.training_num))
        np.random.shuffle(distribution)

        rearranged_inputs = [x[1] for x in sorted(zip(distribution,self.training_inputs))]
        rearranged_outputs = [x[1] for x in sorted(zip(distribution,self.training_outputs))]

        # Determine the total number of all mini batches, which the number of samples is mini_batchsize(50).
        num_complete_minibatches = math.floor(self.training_num / self.mini_batchsize)
        mini_batches = []

        for i in range(num_complete_minibatches):
            mini_batch_inputs = rearranged_inputs[i*self.mini_batchsize : (i+1)*self.mini_batchsize]
            mini_batch_outputs = rearranged_outputs[i*self.mini_batchsize : (i+1)*self.mini_batchsize]
            mini_batch = (mini_batch_inputs, mini_batch_outputs)

            mini_batches.append(mini_batch)

            #If the remaining samples are not enough for a minibatch size(50), the remaining samples are regarded as the last batch.
        if self.training_num % self.mini_batchsize != 0:
            mini_batch_inputs = rearranged_inputs[self.mini_batchsize * num_complete_minibatches:]
            mini_batch_outputs = rearranged_outputs[self.mini_batchsize * num_complete_minibatches:]
            mini_batch = (mini_batch_inputs, mini_batch_outputs)

            mini_batches.append(mini_batch)
        return mini_batches

    def backpropagation_each_epoch(self,mini_batch):
        '''
        Get the total gradient of each mini-batch.
        '''
        mini_batch_grad_b = np.zeros(np.shape(mini_batch[1][0]))
        mini_batch_grad_w = np.zeros((np.shape(mini_batch[1][0])[0],np.shape(mini_batch[0][0])[0]))

        mini_batch_inputs, mini_batch_outputs = mini_batch

        for i in range(len(mini_batch_inputs)):
            grad_b, grad_w = self.backpropagation(mini_batch_inputs[i],mini_batch_outputs[i])
            mini_batch_grad_b = mini_batch_grad_b + grad_b
            mini_batch_grad_w = mini_batch_grad_w + grad_w

        return mini_batch_grad_b, mini_batch_grad_w

    def accuracy(self):
        '''
        By comparing the results of the training set with the test set.
        The accuracy of the neural network is obtained.
        '''
        # Determine the number of samples where the result calculated by the neural network is equal to the actual result.
        correct_number = 0
        for i in range(len(self.testing_inputs)):
            if (self.testing_outputs[i] == self.output_attributes(self.testing_inputs[i])).all():
                correct_number += 1

        return correct_number/len(self.testing_inputs)

    def traning_epoches(self):
        '''
        Iterative training, get weights and biases under the required accuracy.
        '''
        print('Start training the network.')
        seed = 0
        for epochs_num in range(self.epochs_num_max):
            seed += 1
            minibatches = self.get_mini_batches(seed)
            mini_batch_grad_b, mini_batch_grad_w = self.backpropagation_each_epoch(minibatches[0])
            self.biases = self.biases - (self.eta/self.mini_batchsize) * mini_batch_grad_b
            self.weights = self.weights - (self.eta/self.mini_batchsize) * mini_batch_grad_w

            if self.accuracy() > self.target_accuracy:
                break
        print('epochs number is {}\naccuracy is {}'.format(epochs_num+1,self.accuracy()))
        return
