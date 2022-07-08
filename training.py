from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from preprocess import*
from torchdata import*
from loss import*
from network import*
import random

random.seed(1111)

solver = Solver(rna_mats=rna_mats, labels=labels)

#train scMRA model for 200 epochs
for t in range(200):
    print('Epoch: ', t)
    num = solver.train_gcn_adapt(t)
    best_acc = solver.test(t)

ARI = np.around(adjusted_rand_score(np.array(best_acc[1]),np.array(best_acc[0])),5)
AMI = np.around(adjusted_mutual_info_score(np.array(best_acc[1]),np.array(best_acc[0])),5)