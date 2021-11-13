import nn
import dataloader
import pickle

if __name__ == "__main__":
        
    training_data, test_data = dataloader.loaddata()
    nn = nn.Neuralnetwork(sizes=[1600, 2])
    nn.sgd(training_data, test_data)

    pickle.dump(nn, open('neural_network.p', 'wb'))
    print("save neural network. \n")

    
