import questionnaire_evaluation.evaluation_design as evaluation_design

import csv

def csv_read(cvs_path):
    with open(cvs_path,'r') as csvfile:
        csv_data = {}
        for row in csv.DictReader(csvfile):
            for k, v in row.items():
                csv_data.setdefault(k,[]).append(v)
        return csv_data

print('read questionnaire with neural network')
neural_network_file = 'neural_network_minibatch.p'
path_boxes = 'Evaluationshelfer_Daten/boxes'
cvs_path = 'reference/reference_box.csv'
ans_num = len(csv_read(cvs_path).keys())//2
que_num = len(csv_read(cvs_path)['que'])

evaluation_design.Read_questionnaire(neural_network_file, path_boxes, que_num, ans_num).load_results()
