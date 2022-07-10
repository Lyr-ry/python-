from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from preprocess import*
from torchdata import*
from loss import*
from network import*
import random
import csv

random.seed(1111)

#data reading
temp_mat1 = np.array([])
label1 = []
with open('/data/liyiran666/brain/brain/brain5096_train.csv','r') as fp:
    reader=csv.reader(fp)
    for x in reader:
        temp = np.array(x[0:-1])
        temp_mat1 = np.r_[temp_mat1,temp]
        label1 = label1.extend(temp[-1])
        
# temp_mat2 = np.array([])
# label2 = []
# with open('/data/liyiran666/brain/brain/brain3920_train.csv','r') as fp:
#     reader=csv.reader(fp)
#     for x in reader:
#         temp = np.array(x[0:-1])
#         temp_mat1 = np.r_[temp_mat2,temp]
#         label2 = label2.extend(temp[-1])
        
# temp_mat3 = np.array([])
# label3 = []
# with open('/data/liyiran666/brain/brain/brain2904_train.csv','r') as fp:
#     reader=csv.reader(fp)
#     for x in reader:
#         temp = np.array(x[0:-1])
#         temp_mat3 = np.r_[temp_mat3,temp]
#         label3 = label3.extend(temp[-1])
        
# temp_mat4 = np.array([])
# label4 = []
# with open('/data/liyiran666/brain/brain/brain1705_train.csv','r') as fp:
#     reader=csv.reader(fp)
#     for x in reader:
#         temp = np.array(x[0:-1])
#         temp_mat4 = np.r_[temp_mat4,temp]
#         label4 = label4.extend(temp[-1])

temp_mat_t1 = np.array([])
label_t1 = []
with open('/data/liyiran666/brain/brain/brain5096_test.csv','r') as fp:
    reader=csv.reader(fp)
    for x in reader:
        temp = np.array(x[0:-1])
        temp_mat_t1 = np.r_[temp_mat_t1,temp]
        label_t1 = label_t1.extend(temp[-1])
        
rna_mats = [temp_mat1,temp_mat_t1]
labels = [label1,label_t1]

solver = Solver(rna_mats=rna_mats, labels=labels)

#train scMRA model for 200 epochs
for t in range(200):
    print('Epoch: ', t)
    num = solver.train_gcn_adapt(t)
    best_acc = solver.test(t)

ARI = np.around(adjusted_rand_score(np.array(best_acc[1]),np.array(best_acc[0])),5)
AMI = np.around(adjusted_mutual_info_score(np.array(best_acc[1]),np.array(best_acc[0])),5)
