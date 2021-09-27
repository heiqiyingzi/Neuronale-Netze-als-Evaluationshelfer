import os
import numpy as np
import cv2
import pickle
import pandas as pd
import torch

import image_read

class Using_dlnn():
    def __init__(self,que_num,ans_num):
        self.path_boegen = 'Evaluationshelfer_Daten/boegen'
        self.path_boxes = 'Evaluationshelfer_Daten/boxes'
        self.que_num = que_num
        self.ans_num = ans_num
        
    def result_dateup(self,dic,bogen_name,Q_i, A_j,X):
        #parameter = (np.load('dlnn_data/dateup_biases.npy'), np.load('dlnn_data/dateup_weights.npy'))
        dlnn = pickle.load(open("nn_object.p", "rb"))
        a_1 = dlnn.dlnn_crossed(X)
        
        standard = torch.from_numpy(np.array([[1],[0]]))
        a_1 = torch.from_numpy(a_1)
        right_data = a_1.eq(standard.view_as(a_1)).numpy()
        vergleich = np.sum(right_data)//2

        if vergleich:
            if dic[bogen_name][Q_i-1] is None:
                dic[bogen_name][Q_i-1] = str(A_j)
            else:
                dic[bogen_name][Q_i-1]= dic[bogen_name][Q_i-1] + str(A_j)
        return dic

    def load_boxes(self):
        dic={}
        image_data = image_read.image_data(0.75,'Evaluationshelfer_Daten/crosses/work_type_crossed','Evaluationshelfer_Daten/crosses/work_type_empty')
        for file in os.listdir(self.path_boegen):
            bogen_name = os.path.splitext(file)[0]
            curr_file = self.path_boxes +'/' + bogen_name
            if curr_file != 'Evaluationshelfer_Daten/boxes/Bogen1':
                quat_list = [None] * self.que_num
                dic[bogen_name]=quat_list
                for i in range(self.que_num):
                    for j in range(self.ans_num):
                        quat_name = curr_file + '/Q{}A{}.png'.format(i+1,j+1)
                        X = image_data.color_grey(quat_name)
                        dic = self.result_dateup(dic,bogen_name,int(i+1), int(j+1),X)

        dataframe = pd.DataFrame(dic)
        #将DataFrame存储为csv,index表示是否显示行名，default=True
        print('Save the result to result.csv')
        dataframe.to_csv("result.csv",index=False,sep=',')
        return
