import numpy as np
import glob
from PIL import Image

def loaddata():
    print("dataset loading...")
    # crossed 5573, empty 21867
    num_data = 5573 + 21867
    train_num_data = int(round(0.75*num_data))

    image_data = np.zeros(shape=(num_data, 1600), dtype = float)
    crossed_files = glob.glob('Evaluationshelfer_Daten/crosses/work_type_crossed/*.png')
    empty_files = glob.glob('Evaluationshelfer_Daten/crosses/work_type_empty/*.png')
    files = crossed_files + empty_files

    for i in range(len(files)):
        image_data[i] = np.array(Image.open(files[i]).convert('L').getdata())/float(255)
    
    values = np.zeros(shape=(num_data, 2), dtype=float)
    for j in range(num_data):
        if j <= 5573:
            values[j] = np.array([0, 1])
            # crossed
        else:
            values[j] = np.array([1, 0])
            # empty
    
    dataset = [(np.reshape(s, (1600, 1)), np.reshape(y, (2, 1))) for s, y in zip(image_data, values)]

    # np.random.seed(1)
    # np.random.shuffle(dataset)

    crossed_75num = int(round(0.75*5573))
    empty_75num = int(round(0.75*21867))
    
    train_data = dataset[:crossed_75num+1] + dataset[5573:5573+empty_75num-1]
    test_data = dataset[crossed_75num+1:5573] + dataset[5573+empty_75num-1:]
    print("dataset loading completed")

    return train_data, test_data

def loadbogen(bogenpath, num_questions, num_answers):
    bogendata = np.zeros(shape=(num_questions, num_answers, 1600), dtype=float)
    for q in range(14):
        for a in range(5):
            bogendata[q][a] = np.array(Image.open(bogenpath + '/Q{}A{}.png'.format(q+1,a+1)).convert('L').getdata())/float(255)

    return bogendata



