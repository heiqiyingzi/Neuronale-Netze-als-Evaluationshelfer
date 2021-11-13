import dataloader
import pickle
import numpy as np
import csv

if __name__ == "__main__":
    statdict = {}
    for bogen in range(28):
        num_q = 14
        num_a = 5   
        bogen_path = 'Evaluationshelfer_Daten/boxes/Bogen'
        bogendata = dataloader.loadbogen(bogen_path + str(bogen+1), num_q, num_a)   
        nn = pickle.load(open('neural_network.p', 'rb'))

        # eval_res = np.zeros(shape=(num_q, num_a), dtype=int)


        for q in range(num_q):
            ans_q = ''
            for a in range(num_a):
                eval_res = nn.is_crossed(np.reshape(bogendata[q][a], (1600, 1)))
                ans_q += str(eval_res)
            statdict.setdefault('Bogen'+str(bogen+1), []).append(ans_q)

    with open('statistic.csv', 'w') as statfile:
        # headers = []
        # for header in statdict.keys():
        #     header = header.encode()
        #     headers.append(header)
        headers = list(statdict.keys())

        statlist = []
        for q in range(num_q):
            statlist.append([])
            for bogen in range(28):
                statlist[q].append(statdict['Bogen'+str(bogen+1)][q])
        
        statwriter = csv.writer(statfile)
        statwriter.writerow(headers)
        for q in range(num_q):
            statwriter.writerow(statlist[q])
