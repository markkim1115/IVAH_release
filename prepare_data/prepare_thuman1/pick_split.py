import os

subjs = []
with open('THuman_1_train_list.txt', 'r') as f:
    
    subjs += f.readlines()
    subjs = sorted([x.replace('\n','') for x in subjs])
    f.close()

single_subj = []
two_subjs = []
for subj in subjs:
    subj_name = '_'+subj.split('_')[-3]+'_'
    cnt = 0
    subj_dir = []
    for subj_ in subjs:
        if subj_name in subj_:
            cnt += 1
            subj_dir.append(subj_)
    print(subj_name, cnt)

with open('THuman_1_list.txt', 'w') as f:
    for subj in two_subjs:
        f.write(subj+'\n')
    
    for subj in single_subj:
        f.write(subj+'\n')

