import csv
import os
from PIL import Image, ImageDraw
from scipy.optimize import minimize
import numpy as np

class Questionnaire:
    def __init__(self, curr_file, ref_boxes, ref_points, isReference=False, target_path=None):
        self.image = Image.open(curr_file)
        self.init_points_dict = self.get_ref_point_position()
        self.masks = self.init_points_dict['pos']



        if isReference==True:
            self.ref_boxes = self.get_ref_box_position()
            self.ref_points = self.get_mask_positon()

            self.que_num = int(len(self.ref_boxes['que']))
            self.ans_num = int((len(self.ref_boxes.keys())-1)/2)

            self.curr_boxes = self.ref_boxes

            self.target_path = target_path
            self.extract_boxes()

        else:
            self.ref_boxes = ref_boxes
            self.curr_boxes = {}
            
            self.ref_points = ref_points
            self.curr_points = self.get_mask_positon()

            self.que_num = int(len(self.ref_boxes['que']))
            self.ans_num = int((len(self.ref_boxes.keys())-1)/2)

            self.trans_matrix = self.get_trans_matrix()

            self.transformisDone = False
            self.boxes_transformed()

            self.target_path = target_path
            self.extract_boxes()


    def get_ref_box_position(self)  ->dict:
        with open('reference/reference_box.csv') as positionfile:
            reader = csv.DictReader(positionfile)
            ref_box_positions = {}
            for row in reader:
                for k, v in row.items():
                    if k=='que':
                        ref_box_positions.setdefault(k,[]).append(v)
                    else:
                        ref_box_positions.setdefault(k,[]).append(int(v))
        return ref_box_positions


    def get_ref_point_position(self)  ->dict:
        ref_point_positions = {}
        with open('reference/reference_point.csv') as positionfile:
            reader = csv.DictReader(positionfile)
            ref_point_positions = {}
            for row in reader:
                for k, v in row.items():
                    if k=='pos':
                        ref_point_positions.setdefault(k,[]).append(v)
                    else:
                        ref_point_positions.setdefault(k,[]).append(int(v))
        return ref_point_positions

    def get_mask_positon(self):
        mask_positions = []
        for i, mask in enumerate(self.masks):
            mask_image = Image.open('Evaluationshelfer_Daten/masks/{}.png'.format(mask))
            search_area_curr = [self.init_points_dict['x'][i]-50,
                                self.init_points_dict['y'][i]-50,
                                self.init_points_dict['x'][i]+50,
                                self.init_points_dict['y'][i]+50,]
            mask_curr_size = list(mask_image.size)
            mask_pos = self.get_position(mask_image, search_area_curr, mask_curr_size)
            mask_positions.append(mask_pos)
        return mask_positions

    def get_position(self, mask, search_area:list, mask_size):
        # search_area:[x1, y1, x2, y2]
        # mask_size:[x_size, y_size]
        pos = []
        mse_init = 99999
        mask_img = mask.load()

        for x in range(search_area[0],search_area[2]):
            for y in range(search_area[1],search_area[3]):
                # extract rectangle being searched 
                cropped_area = self.image.crop([x, y, x+mask_size[0], y+mask_size[1]]).load()
                # caculate mean squared error between cropped area and current mask
                mse_err = self.get_mse_error(cropped_area, mask_img, x_max = mask_size[0], y_max=mask_size[1])

                if mse_err < mse_init:
                    mse_init = mse_err
                    pos = [x, y]

        return pos


    def get_mse_error(self, img1, img2, x_max, y_max):
        mse = 0
        for i in range(x_max):
            for j in range(y_max):
                mse += (img1[i,j] - img2[i,j])**2
        mean_mse = mse/(x_max*y_max)
        return mean_mse

    
    

    def get_trans_matrix(self):

        num_of_points = len(self.ref_points)
        C = np.zeros((2 * num_of_points, 6))
        d = np.zeros((2 * num_of_points, 1))
        index = 0
        for i, pos_mask in enumerate(self.ref_points):
            C[index][0] = pos_mask[0]
            C[index][1] = pos_mask[1]
            C[index][4] = 1
            C[index + 1][2] = pos_mask[0]
            C[index + 1][3] = pos_mask[1]
            C[index + 1][5] = 1
            d[index][0] = self.curr_points[i][0]
            d[index + 1][0] = self.curr_points[i][1]
            index += 2
        best_fit = np.linalg.lstsq(C, d, rcond=None)[0]
        return best_fit.T[0]




    def transform(self, ref_point):
        A = self.trans_matrix[:4].reshape((2, 2))
        b = self.trans_matrix[4:]
        new = np.dot(A, ref_point) + b
        return new


    def boxes_transformed(self):
        if self.transformisDone == False:
            for j in range(self.ans_num):
                for i in range(self.que_num):
                    ref_ij = [self.ref_boxes['x'+str(j+1)][i], self.ref_boxes['y'+str(j+1)][i]]
                    new_ij = self.transform(ref_ij)
                    self.curr_boxes.setdefault('x'+str(j+1), []).append(new_ij[0])
                    self.curr_boxes.setdefault('y'+str(j+1), []).append(new_ij[1])
            self.transformisDone = True   


    def extract_boxes(self):
        for i in range(self.que_num):
            for j in range(self.ans_num):
                box = self.image.crop([self.curr_boxes['x'+str(j+1)][i]-30, 
                                        self.curr_boxes['y'+str(j+1)][i]-30,
                                        self.curr_boxes['x'+str(j+1)][i]+30, 
                                        self.curr_boxes['y'+str(j+1)][i]+30])
                box = box.resize((50, 50))
                box.save(self.target_path + '/Q{}A{}.png'.format(i+1,j+1))