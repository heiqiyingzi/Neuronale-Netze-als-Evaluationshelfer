import numpy as np
import math
import torch



class MyNeuralNetwork(object):
    def __init__(self,sizes):
        self.sizes = sizes
        #(1600x2)
        # layers
        self.num_layers = len(sizes)
        self.inputs_num = sizes[0]

        np.random.seed(0)
        self.weights = (np.random.normal(0, 1,(self.sizes[1],self.sizes[0]))) /math.sqrt(self.sizes[0])
        #(2x1600)
        self.biases =np.random.normal(0, 1, [self.sizes[1],1])
        #(2x1)
        self.parameter = (self.biases,self.weights)
        # batch, iterations and learning rate
        self.mini_batchsize = 50
        self.epochs = 5
        self.eta = 0.1
        
    def sigmoid(self,z):
        # Our activation function: f(x) = 1 / (1 + e^(-x))
        #输出(2x1)
        return 1/(1 + np.exp(-z))

    def feedforward(self,X):
        # inputs_X(1600x50),weights_i, biases_i
        # Weight inputs, add bias, then use the activation function
        z = np.dot(self.weights, X) + self.biases
        # (2x1600)x(1600x50)+(2x1)
        a_1 = self.sigmoid(z)
        #输出a1是(2x50)
        return a_1

    def backward_propagation(self,X,Y):
        # X=(1600x50);Y=(2x50);weights_i=(2x1600); biases_i=(2x1)
        grad_b = self.feedforward(X) - Y
        # grad_b=(2x50)-(2x50)=(2x50)
        sum_grad_b0 = np.sum(grad_b, axis=1)
        num_sum_grad_b0 = len(sum_grad_b0)
        sum_grad_b = sum_grad_b0.reshape((num_sum_grad_b0,1))
        # sum_grad_b =(2x1)
        sum_grad_w = np.dot(grad_b, np.transpose(X))
        # sum_grad_w = (2x50)x(50x1600) = (2x1600)
        grads = (sum_grad_b,sum_grad_w)
        return grads

    def costfunction(self, a_1, Y):
        #(2x50), (2x50)
        num = a_1.shape[1]
        sum_cost = np.zeros((a_1.shape[0]))
        for i in range(num):
            cost = -Y[:,i]*np.log(a_1[:,i])-(1-Y[:,i])*np.log(1-a_1[:,i])
            sum_cost =sum_cost + cost
        return np.sum(sum_cost)

    def update_parameters_with_gd(self,Y,grads):
        sum_grad_b,sum_grad_w = grads
        self.biases = self.biases - (self.eta/self.mini_batchsize)*sum_grad_b
        self.weights = self.weights - (self.eta/self.mini_batchsize)*sum_grad_w
        #(2x1600), (2x1)
        return (self.biases,self.weights)

    def random_mini_batches(self,training_data,seed=1):
        n = training_data.shape[0]-2
        m = training_data.shape[1]
        X = training_data[:n,:]
        Y = training_data[n:(n+2),:]
        # '输入：X的维度是（n,m），m是样本数，n是每个样本的特征数'
        # (1600, 20579), (2, 20579)
        
        np.random.seed(seed)
        #之后生成的随机数列固定
        mini_batches = []
        #step1：打乱训练集
        #生成0~m-1随机顺序的值，作为我们的下标
        permutation = list(np.random.permutation(m))
        #得到打乱后的训练集
        shuffled_X = X[:,permutation]
        #把x的每行的元素的顺序打乱
        shuffled_Y = Y[:,permutation]

        #step2：按照batchsize分割训练集
        #得到总的子集数目，math.floor表示向下取整
        num_complete_minibatches = math.floor(m / self.mini_batchsize)
        # 27000/50
        mini_batches = []
        for k in range(0,num_complete_minibatches):
            #冒号：表示取所有行，第二个参数a：b表示取第a列到b-1列，不包括b
            mini_batch_X = shuffled_X[:, k * self.mini_batchsize:(k+1) * self.mini_batchsize]
            #(1600x50)
            mini_batch_Y = shuffled_Y[:, k * self.mini_batchsize:(k+1) * self.mini_batchsize]
            #(2x50)
            mini_batch = np.append(mini_batch_X,mini_batch_Y,axis=0)
            #array(1602x50)
            mini_batches.append(mini_batch)
            #len(mini_batches)=411

            #m % self.mini_batchsize != 0表示还有剩余的不够一个batch大小，把剩下的作为一个batch
        if m % self.mini_batchsize != 0:
            mini_batch_X = shuffled_X[:,self.mini_batchsize * num_complete_minibatches:]
            #(1600x余数)
            mini_batch_Y = shuffled_Y[:,self.mini_batchsize * num_complete_minibatches:]
            #(2x余数)
            mini_batch = np.append(mini_batch_X,mini_batch_Y,axis=0)
            mini_batches.append(mini_batch)

        return mini_batches
            # len(mini_batches)=412

    def dlnn_traning(self,training_data,test_data):
        # (1600, 20579) (2, 20579)
        seed = 0
        for i in range(self.epochs):
            seed = seed + 1
            minibatches = self.random_mini_batches(training_data,seed)
            for minibatch in minibatches:
                minibatch_X = minibatch[:self.inputs_num,:]
                minibatch_Y = minibatch[self.inputs_num:,:]
                #取出batch中的 X(1600x50),Y(2x50)
                #cache 为临时的w,b
                # a_1 = self.feedforward(minibatch_X)
                # cost_value = self.costfunction(a_1, minibatch_Y)
                grads = self.backward_propagation(minibatch_X,minibatch_Y)
                self.parameter = self.update_parameters_with_gd(minibatch_Y, grads)
                #循环后得到1次ecope迭代结果
                accuracy = self.accuracy(test_data)
                if accuracy>0.97:
                    break
                epochs =i+1
        
        return self.parameter,epochs,accuracy 

    def dlnn_crossed(self,X):
        a_1 = self.feedforward(X)
        n = a_1.shape[0]
        m = a_1.shape[1]
        a_1_prediction = np.zeros((n,m))
        for i in range(m):
            maxindex_a_1i = np.argmax(a_1[:,i], axis=0)#返回行号
            a_1_prediction[:,i][maxindex_a_1i] = 1
            if a_1[:,i][0] == a_1[:,i][1]:
                a_1_prediction[:,i][1] = 1
            
        return a_1_prediction

    def accuracy(self,test_data):
        X_test = test_data[:self.inputs_num,:]
        Y_test = test_data[self.inputs_num:,:]
        # predictions(2x  ), labels(2x  )
        prediction = self.dlnn_crossed(X_test)
        row_num = prediction.shape[0]
        col_num = prediction.shape[1]

        Y_test = torch.from_numpy(Y_test)
        prediction = torch.from_numpy(prediction)
        right_data = prediction.eq(Y_test.view_as(prediction)).numpy()

        accuracy_col = np.sum(right_data,axis=0)//row_num
        accuracy = np.sum(accuracy_col)/col_num

        return accuracy
