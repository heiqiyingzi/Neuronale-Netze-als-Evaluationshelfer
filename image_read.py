import numpy as np
import math
from PIL import Image
import os

class image_data():
    def __init__(self,training_percent,path_crossed,path_empty):
        self.training_percent = training_percent
        self.path_crossed = path_crossed
        self.path_empty = path_empty
        
    def color_grey(self,imgfile):
        img_gray = np.array(Image.open(imgfile).convert('L'), 'f')
        cols,rows = img_gray.shape # 图像大小
        
        img_array = 1-img_gray/float(255)
        Value = img_array.reshape(cols*rows,1)
        #(1600,1)
        return Value
    
    def read_foder(self,directory_name, output_value):
        #output_value=array(2x1)
        # this loop is for read each image in this foder,directory_name is the foder name with images.
        files = os.listdir(r""+directory_name)   # 读入文件夹
        num_png = len(files)
        i = 0

        for filename in files:
            i += 1
            img = self.color_grey(directory_name + "/" + filename)
            #img = array(1600x1)
            if i==1:
                sum_img = img
                sum_outputs = output_value
            else:
            # this if for store all of the image data
            # this function is for read image,the input is directory name
            # img is used to store the image data
                sum_img = np.append(sum_img, img,axis=1)
                #array(1600,5573+)
                sum_outputs = np.append(sum_outputs,output_value,axis=1)
                #array(2,5573+)

        training_num = int(math.floor(num_png * self.training_percent))
        img_training = sum_img[:,:training_num]
        outputs_training = sum_outputs[:,:training_num]
        training_data = np.append(img_training,outputs_training,axis=0)
        #(1602, 5573*0.75)

        img_test = sum_img[:,training_num:]
        outputs_test = sum_outputs[:,training_num:]
        test_data = np.append(img_test, outputs_test,axis=0)
        #(1602, 5573*0.25)
    
        return training_data ,test_data  #文件夹中所有图像的个数=len(array_of_img)

    def load_foder(self):
        training_data_crossed, test_data_crossed = self.read_foder(self.path_crossed,np.array([[1],[0]]))
        print('Image data reading for training completed.')
        training_data_empty, test_data_empty = self.read_foder(self.path_empty,np.array([[0],[1]]))
        print('Image data reading for testing completed.')
        
        training_data = np.append(training_data_crossed,training_data_empty,axis=1)
        test_data = np.append(test_data_crossed,test_data_empty,axis=1)
        
        print('Save data to dlnn_data/image_data_npy/training_data.npy and test_data.npy')
    
        path = 'dlnn_data/image_data_npy'
        # 创建文件夹
        if not os.path.exists(path):
            os.makedirs(path)  
        np.save("dlnn_data/image_data_npy/training_data.npy",training_data)
        np.save("dlnn_data/image_data_npy/test_data.npy",test_data)
        return
