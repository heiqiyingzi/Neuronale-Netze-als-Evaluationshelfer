from PIL import Image
import numpy as np
import os
import math

class Image_input_grayvalue():
    def __init__(self,folder_path):
        self.folder_path = folder_path

    def Inverse_conversion(self,imgfile):
        '''
        invert numpy-array "image"(normalize) to [1, 0]
        '''
        Converted_grayvalue = np.array(Image.open(imgfile).convert('L'))/float(255)

        return 1-Converted_grayvalue

    def get_grayvalue_images(self,imgfile):
        '''
        Get the gray value of an image
        (the gray value is normalized and inverted)
        '''
        value_image_array = self.Inverse_conversion(imgfile)
        image_reshape_array = value_image_array.reshape(1600,1)

        return image_reshape_array

    def get_grayvalue_all_images(self):
        '''
        Get the gray value of all images in the specified folder
        (the gray value is normalized and inverted)
        '''
        files = os.listdir(r''+self.folder_path)
        image_array_sum = []
        for filename in files:
            file_path = self.folder_path+ '/' + filename
            image_array = self.get_grayvalue_images(file_path)
            image_array_sum.append(image_array)
        return image_array_sum

class Image_output_result():
    '''
    Assign to the result list according to the folder attribute (crossed or empty)
    '''
    def __init__(self,folder_path, folder_attribute):
        self.folder_path = folder_path
        # folder_attribute crossed: array([[0],[1]])
        # folder_attribute empty: array([[1],[0]])
        self.folder_attribute = folder_attribute

    def get_all_images_output_result(self):
        '''
        Assign all images in the folder attributes to the result list: image_output_result_sum
        '''
        files = os.listdir(r''+self.folder_path)
        image_output_result_sum = []
        for filename in files:
            file_path = self.folder_path+ '/' + filename
            image_output_result_sum.append(self.folder_attribute)
        return image_output_result_sum


class get_training_and_testing_data():
    '''
    According to the training percent, the input and output of the training and testing data are obtained.
    '''
    def __init__(self, training_percent, path_crossed, path_empty):
        self.training_num = training_percent
        self.path_crossed = path_crossed
        self.crossed_folder_attribute = np.array([[0],[1]])
        self.path_empty = path_empty
        self.empty_folder_attribute = np.array([[1],[0]])


    def get_images_inputs(self,path_folder):
        '''
        Determine the inputs of images and the number of training data according to the folder.
        '''
        images_inputs = Image_input_grayvalue(path_folder).get_grayvalue_all_images()
        training_num = int(math.floor(len(images_inputs)*self.training_num))
        return images_inputs, training_num

    def get_images_outputs(self,path_folder,folder_attribute):
        '''
        Determine the outputs of images and the number of training data according to the folder
        '''
        images_outputs = Image_output_result(path_folder,folder_attribute).get_all_images_output_result()
        training_num = int(math.floor(len(images_outputs)*self.training_num))

        return images_outputs, training_num

    def get_training_data(self):
        '''
        Integrate the training data of all the files in the folder.
        '''
        crossed_images_inputs, crossed_training_num = self.get_images_inputs(self.path_crossed)
        crossed_images_outputs, crossed_training_num = self.get_images_outputs(self.path_crossed,self.crossed_folder_attribute)

        empty_images_inputs, empty_training_num = self.get_images_inputs(self.path_empty)
        empty_images_outputs, empty_training_num = self.get_images_outputs(self.path_empty,self.empty_folder_attribute)

        training_inputs = crossed_images_inputs[: crossed_training_num] + empty_images_inputs[: empty_training_num]
        training_outputs = crossed_images_outputs[: crossed_training_num] + empty_images_outputs[: empty_training_num]

        print('Image data for training completed.')

        return training_inputs, training_outputs

    def get_testing_data(self):
        '''
        Integrate the testing data of all the files in the folder.
        '''
        crossed_images_inputs, crossed_training_num = self.get_images_inputs(self.path_crossed)
        crossed_images_outputs, crossed_training_num = self.get_images_outputs(self.path_crossed,self.crossed_folder_attribute)

        empty_images_inputs, empty_training_num = self.get_images_inputs(self.path_empty)
        empty_images_outputs, empty_training_num = self.get_images_outputs(self.path_empty,self.empty_folder_attribute)

        testing_inputs = crossed_images_inputs[crossed_training_num :] + empty_images_inputs[empty_training_num :]
        testing_outputs = crossed_images_outputs[crossed_training_num :] + empty_images_outputs[empty_training_num :]

        print('Image data for testing completed.')

        return testing_inputs, testing_outputs

    def load_data(self):
        '''
        Save the training data and test data to a specific location.
        '''
        print('Loading...')
        training_inputs, training_outputs = self.get_training_data()
        training_list = training_inputs + training_outputs

        testing_inputs, testing_outputs = self.get_testing_data()
        testing_list = testing_inputs + testing_outputs
        print('Completed\n')

        return training_list, testing_list
