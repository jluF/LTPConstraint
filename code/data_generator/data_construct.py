import numpy as np
import pandas as pd
import os
File_dir = "/media/disk/swzxf/Contrast_experiment/LTPConstraint/5SrRNA/CT_FILE_Un_80/"
Save_dir = "/media/disk/swzxf/PycharmProjects/test/data/128/"
word_coder = {
    "A":1,
    "U":2,
    "G":3,
    "C":4,
}
def generate_dataset(original_list, limit_length):
    seq = []
    label = []
    for i in range(limit_length):
        temp = []
        for j in range(limit_length):
            temp.append(0)
        label.append(temp)
    for ele in original_list:
        seq.append(word_coder.get(ele[1], 0))
        if(ele[4]!=0):
            x = ele[0]-1
            y = ele[4]-1
            label[x][y]=1
            label[y][x]=1
    for i in range(len(original_list), limit_length):
        seq.append(0)
    return seq, label
seq_list = []
label_atrix = []
limit_length = 128
kk=0
for f_name in os.listdir(File_dir):
    print('now processing:'+str(kk))
    kk = kk+1
    ele = pd.read_csv(File_dir+f_name,sep="\s+",skiprows=1, header=None)
    ele = ele.values.tolist()
    if len(ele) <= 128:
        _seq, _label = generate_dataset(ele, limit_length)
        seq_list.append(_seq)
        label_atrix.append(_label)
print('now traslating')
seq_list = np.array(seq_list)
label_atrix = np.array(label_atrix, dtype=np.float32)

print(seq_list.shape, label_atrix.shape)

from numpy import savez, loadtxt
np.savez(Save_dir+'5SrRNA_Un_80', seq_list, label_atrix)