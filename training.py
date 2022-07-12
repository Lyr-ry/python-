from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from preprocess import*
from torchdata import*
from loss import*
from network import*
import random
import csv
import matplotlib.pyplot as plt

# import module related to tensionboard for generate working journal
from torch.utils.tensorboard import SummaryWriter

# creat working journal and store at assigned path
writer = SummaryWriter('/home/qukun/dingyiemail/lyr/output/logs')

random.seed(1111)

#data reading
temp_mat1 = []
label1 = []
with open('/home/qukun/dingyiemail/lyr/brain/brain5096_train.csv','r') as fp:
    tempstr = fp.readlines()
    for i in range(len(tempstr)-1):
        temp_mat1.append([])
        x = tempstr[i+1].split(',')
        for j in range(len(x)-1):
            temp_mat1[i].append(x[j])
        if i == len(tempstr)-2:
            labelstr = x[-1]
        else:
            labelstr = x[-1][0:-1]
        label1.append(labelstr)
temp1 = np.array(temp_mat1)
        

temp_mat2 = []
label2 = []
with open('/home/qukun/dingyiemail/lyr/brain/brain5096_test.csv','r') as fp:
    tempstr = fp.readlines()
    for i in range(len(tempstr)-1):
        temp_mat2.append([])
        x = tempstr[i+1].split(',')
        for j in range(len(x)-1):
            temp_mat2[i].append(x[j])
        if i == len(tempstr)-2:
            labelstr = x[-1]
        else:
            labelstr = x[-1][0:-1]
        label2.append(labelstr)
temp2 = np.array(temp_mat2)
        
rna_mats = [temp1,temp2]
labels = [label1,label2]


solver = Solver(rna_mats=rna_mats, labels=labels)


#train scMRA model for 100 epochs
for t in range(100):
    print('Epoch: ', t)
    num = solver.train_gcn_adapt(t)
    best_acc,loss = solver.test(t)
    #update the Loss and Accuracy
    writer.add_scalar('Accuracy', best_acc, t)
    writer.add_scalar('Loss', loss, t)

    
ARI = np.around(adjusted_rand_score(np.array(best_acc[1]),np.array(best_acc[0])),5)
AMI = np.around(adjusted_mutual_info_score(np.array(best_acc[1]),np.array(best_acc[0])),5)