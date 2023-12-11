from Jtools import *

f = open('data/ML_txt_data/total.txt', 'r')
data = f.readlines()
print(f'Total data volume: {len(data)}')
label_0 = 0
label_1 = 0
for line in tqdm(data):
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split('\t')
        label_1 += int(words[3])
label_0 = len(data) - label_1
print(f'Label 0: {label_0}, Label 1: {label_1}')
f.close()