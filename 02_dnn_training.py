import dnn_training.dnn_resource as dnn_resource
import dnn_training.dnn as dnn

import pickle

training_percent = 0.75
path_crossed = 'Evaluationshelfer_Daten/crosses/work_type_crossed'
path_empty = 'Evaluationshelfer_Daten/crosses/work_type_empty'
NeuralNetwork_sizes = (1600,2)
target_accuracy = 0.97

# Read the training set data
training_data, testing_data = dnn_resource.get_training_and_testing_data(training_percent, path_crossed, path_empty).load_data()
#Training neural network
nn = dnn.NeuralNetwork_training(NeuralNetwork_sizes,training_data, testing_data,target_accuracy)
nn.traning_epoches()
# save neural network
pickle.dump(nn, open('neural_network_minibatch.p', 'wb'))
print("Saved neural network to file.\n")
