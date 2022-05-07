import os
import pandas as pd
import numpy as np
File_dir = '/media/disk/swzxf/Contrast_experiment/pseudoknots/512_CT_FILE/'
Save_dir = '/media/disk/swzxf/PycharmProjects/test/data/pseudoknots/'
word_coder = {
    "A":1,
    "U":2,
    "G":3,
    "C":4,
}
# 获取长度为length的初始全0二维矩阵和一维矩阵
def get_original_matrix(length):
    node_i = []
    seq = []
    for i in range(length):
        seq.append(0)
        node_j = []
        for j in range(length):
            node_j.append(0)
        node_i.append(node_j)
    return node_i, seq
# 使用ct文件生成二维关系矩阵和输入的序列一维矩阵
def generate_ct_relation_matrix(original_list, length):
    label, sequence = get_original_matrix(length)
    ih = 0
    for ele in original_list:
        sequence[ih] = word_coder.get(ele[1], 0)
        ih = ih+1
        if(ele[4]!=0):
            x = ele[0]-1
            y = ele[4]-1
            if x < len(original_list) and y < len(original_list):
                label[x][y]=1
                label[y][x]=1
    return sequence, label
# 512的生成步骤
seq_512, label_512 = [], []
for name in os.listdir(File_dir):
    if name.split('_')[1]=='CRW':continue
    df = pd.read_csv(File_dir+name.strip(),sep="\s+",skiprows=1, header=None)
    df = df.values.tolist()
    length = len(df)
    if length > 512: continue
    print('start:'+name)
    if length <= 512 and length > 128:
        seq, label = generate_ct_relation_matrix(df, 512)
        seq_512.append(seq)
        label_512.append(label)
    print('end:'+name)
seq_512 = np.array(seq_512)
label_512 = np.array(label_512, dtype=np.float32)
print(seq_512.shape, label_512.shape)
save_name = 'pseudoknot_512'

np.savez(Save_dir+save_name, seq_512, label_512)
