import csv
import pandas as pd
import image_result


if __name__ == "__main__":

    with open('reference/reference_box.csv', 'r') as file:
        data = file.readlines()
        ans_num = len((data[0]).split(","))//2
        que_num = len(data)-1
        image_re = image_result.Using_dlnn(que_num,ans_num)
        image_re.load_boxes()

        result_data = pd.read_csv('result.csv')  # 读取训练数据
        # (14, 27)
        print(result_data)
