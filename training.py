from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from preprocess import*
from torchdata import*
from loss import*
from network import*
import random
import csv
import matplotlib.pyplot as plt

random.seed(1111)

#data reading
temp_mat1 = np.array([])
label1 = []
with open('/home/qukun/dingyiemail/lyr/brain/brain5096_train.csv','r') as fp:
    reader=csv.reader(fp)
    for x in reader:
        temp = np.array(x[0:-1])
        temp_mat1 = np.r_[temp_mat1,temp]
        label1.extend(temp[-1])
        
temp_mat_t1 = np.array([])
label_t1 = []
with open('/home/qukun/dingyiemail/lyr/brain/brain5096_test.csv','r') as fp:
    reader=csv.reader(fp)
    for x in reader:
        temp = np.array(x[0:-1])
        temp_mat_t1 = np.r_[temp_mat_t1,temp]
        label_t1.extend(temp[-1])
        
rna_mats = [temp_mat1,temp_mat_t1]
labels = [label1,label_t1]

#gain the value of nclasses
n = len(set(label1))

solver = Solver(rna_mats=rna_mats, labels=labels,nclasses=n)

acc_l = []
loss_l = []
x_l = []

#train scMRA model for 200 epochs
for t in range(50):
    print('Epoch: ', t)
    num = solver.train_gcn_adapt(t)
    best_acc,loss = solver.test(t)
    #update the Loss and Accuracy of train and validate
    acc_l.extend(best_acc)
    loss_l.extend(loss)
    x_l.extend(t)

#display the curves of best_accuracy and loss
plt.figure(1)
plt.scatter(x_l,acc_l,label='best_acc')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(loc="best")
plt.savefig("/home/qukun/dingyiemail/lyr/output/best_accuracy.png")
plt.figure(2)
plt.scatter(x_l,loss_l,label='loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(loc="best")
plt.savefig("/home/qukun/dingyiemail/lyr/output/loss.png")
    
ARI = np.around(adjusted_rand_score(np.array(best_acc[1]),np.array(best_acc[0])),5)
AMI = np.around(adjusted_mutual_info_score(np.array(best_acc[1]),np.array(best_acc[0])),5)