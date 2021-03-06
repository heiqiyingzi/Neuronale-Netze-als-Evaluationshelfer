import matplotlib.pyplot as plt
import numpy as np
import csv

if __name__ == "__main__":
    # import data

    with open('statistic.csv') as resultfile:
        result = csv.reader(resultfile)
        result_list = []
        for row in result:
            result_list.append(row)

    result_list = result_list[1:]


    # prepare data

    def count(list):
        count_res = np.zeros(5)
        for ans in list:
            for i in range(5):
                count_res[i] += int(ans[i])
        return count_res


    count_list = np.zeros(shape=(14, 5))
    total_count_list = np.zeros(shape=(1, 5))

    for que in range(14):
        count_list[que] = np.array(count(result_list[que]))
        total_count_list += count_list[que]




    labels =['++', '+', '0', '-', '--']


    # plot bars

    fig = plt.figure(figsize=(8, 5))

    plt.bar(range(5), count(total_count_list), width=0.6, tick_label= labels)
    plt.title('Overview of distribution', fontsize=12)
    plt.ylabel('amount of answers', fontsize=10)
    for a,b in zip(range(5),count(np.array(total_count_list))):
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=7)


    fig = plt.subplots(2, 7,figsize=(18, 8))
    plt.subplots_adjust(wspace =0.8, hspace =0.5)

    for i in range(14):
        plt.subplot(2, 7, i+1)
        plt.bar(range(5), count_list[i], width=0.5, tick_label= labels)

        for a,b in zip(range(5),count_list[i]):
            plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=5.5)
            plt.title('Questionnaire '+str(i+1), fontsize=12)
            plt.ylabel('amount of answers', fontsize=10)
    plt.suptitle('Distribution of answers ', fontsize=20) 



    # plot pies

    fig = plt.figure(figsize=(8, 8))
    plt.pie(count(total_count_list), labels=labels, autopct='%.1f%%', radius=0.9)
    plt.title('Overview of percentage', fontsize=12)


    fig = plt.figure(figsize=(18, 10))
    plt.subplots_adjust(wspace =0.8, hspace =1)

    for i in range(14):
        ax = fig.add_subplot(3, 5, i+1)
        plt.pie(count_list[i], labels=labels, autopct='%.1f%%', radius=1.8)
        plt.title('Question '+str(i+1), fontsize=10)
    plt.suptitle('Percentage of answers ', fontsize=20)      



    plt.show()

