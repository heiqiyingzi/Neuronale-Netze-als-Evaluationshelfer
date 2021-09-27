import numpy as np
import math
import pickle

import image_read
import dlnn as nn

if __name__ == "__main__":
    # load training data
    training_percent = 0.75
    path_crossed = 'Evaluationshelfer_Daten/crosses/work_type_crossed'
    path_empty = 'Evaluationshelfer_Daten/crosses/work_type_empty'
    image_data = image_read.image_data(training_percent,path_crossed,path_empty)
    image_data.load_foder()
    # data needs to be in same folder as script
    
    training_data = np.load("dlnn_data/image_data_npy/training_data.npy")
    test_data = np.load("dlnn_data/image_data_npy/test_data.npy")

    print('The image data has been loaded.\nStart training the network.')

    # actual training of the NN
    evalnn = nn.MyNeuralNetwork([1600, 2])
    dateup_parameter,epochs,accuracy = evalnn.dlnn_traning(training_data,test_data)
    dateup_biases, dateup_weights = dateup_parameter
    
    print('Save biases and weights to dlnn_data/image_data_npy/training_data.npy and test_data.npy')
    np.save("dlnn_data/dateup_biases.npy",dateup_biases)
    np.save("dlnn_data/dateup_weights.npy",dateup_weights)
    print('epochs:',epochs)
    print('accuracy:',accuracy)

    # save nn via pickle
    pickle.dump(evalnn, open("nn_object.p", "wb"))
    print("Saved neural network to file.")
