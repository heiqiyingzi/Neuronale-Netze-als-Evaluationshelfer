import pickle
import pandas as pd
from PIL import Image
import numpy as np
import csv
import os

class Use_neural_network():
    '''
    Use the parameters in the neural network.
    '''
    def __init__(self, neural_network_file):
        self.neural_network_file = neural_network_file

    def get_outputs(self,inputs):
        '''
        Load the neural network. Use it to get the images attributes(output value).
        '''
        dnn_data = pickle.load(open(self.neural_network_file, "rb"))
        # dnn = dnn_data.NeuralNetwork_basic_formula()
        return dnn_data.outputs_attributes(inputs)


class Read_questionnaire(Use_neural_network):
    '''
    Read all screenshots of the questionnaire and judge (crossed or not) the results.
    Sort it into a dictionary and format it in csv format.
    '''
    def __init__(self,neural_network_file, path_boxes, que_num, ans_num):
        Use_neural_network.__init__(self,neural_network_file)
        self.que_num = que_num
        self.ans_num = ans_num
        self.path_boxes = path_boxes

        # self.folder_path is boxes_path
        self.bogens_path,self.bogens_name = self.read_boxs_folder()


    def read_boxs_folder(self)->list:
        '''
        Get all bogen_folder paths in boxes_folder and than resort.
        for exmaple: Bogen1, Bogen2,...
        '''
        bogens = os.listdir(r''+self.path_boxes)
        bogens_path = []
        bogens_name = []
        # self.folder_path is boxes_path
        for bogen_i in range(len(bogens)):
            bogen_path = self.path_boxes+'/'+'Bogen{}'.format(bogen_i+1)

            bogens_path.append(bogen_path)
            bogens_name.append('Bogen{}'.format(bogen_i+1))
        return bogens_path, bogens_name

    def get_grayvalue_images(self)->list:
        '''
        Get the gray value of an image for all bogens
        (the gray value is normalized and inverted)
        Take each question of each questionnaire as the smallest unit.
        '''
        bogens_path,_ = self.read_boxs_folder()
        bogens_images = []
        for bogen_num in range(len(bogens_path)):
            ques_images = []
            for que_i in range(self.que_num):
                que_images = []
                for ans_i in range(self.ans_num):
                    image_path = bogens_path[bogen_num]+'/'+'Q{}A{}.png'.format(que_i+1, ans_i+1)
                    value_image_array = 1-np.array(Image.open(image_path).convert('L'))/float(255)
                    image_reshape_array = value_image_array.reshape(1600,1)
                    que_images.append(image_reshape_array)
                ques_images.append(que_images)
            bogens_images.append(ques_images)
        return bogens_images

    def get_bogens_attribute(self):
        '''
        Call the neural network parameters
        and calculate the properties of each box in the questionnaire.
        '''
        bogens_inputs = self.get_grayvalue_images()
        return Use_neural_network.get_outputs(self,bogens_inputs)

    def get_result_all_bogens(self) ->list:
        '''
        Get the results of all images in the subfolder(bogen_name).
        Judging by the attribute of the box.
        if the attribute is a crossed(array([[1],[0])), the corresponding answer number will be stored.
        '''
        # get the all images_path of subfolder(z.B. Bogen1)
        bogens_result = []
        for bogen_attribute in self.get_bogens_attribute():
            ques_result  = ['' for _ in range(self.que_num)]
            for que_i in range(self.que_num):
                answerts_result = ''
                for ans_i in range(self.ans_num):
                    if (bogen_attribute[que_i][ans_i] == np.array([[0],[1]])).all():
                        answerts_result += str(ans_i+1)
                ques_result[que_i] = answerts_result
            bogens_result.append(ques_result)
        return bogens_result

    def results_to_dic(self):
        '''
        To make a dictionary.
        The key is the name of the bogen, and the contents are the results of the bogen.
        '''
        questionnaire_results_dic = {}
        for bogen_i in range(len(self.bogens_name)):
            questionnaire_results_dic[self.bogens_name[bogen_i]] = self.get_result_all_bogens()[bogen_i]
            print('{} completed'.format(self.bogens_name[bogen_i]))

        return questionnaire_results_dic

    def load_results(self):
        '''
        Convert the dictionary results into csv format and save.
        '''
        result_data = pd.DataFrame(self.results_to_dic())
        print('Save the result to result.csv')
        result_data.to_csv('result.csv',index=False,sep=',')
        return
