#!/usr/bin/env python
# coding: utf-8

# In[23]:


import json

train_folds = {}
for i in range(4):
    filename = f'fold-{i}'
    with open(filename, 'rt') as f:
        for l in f.readlines():
            d = json.loads(l)
            train_folds.update(d)

len(train_folds)

with open("pretrain_split.json", 'wt') as f:
    json.dump(train_folds, f)

with open('fold-4', 'rt') as f:
    cv_dict = json.load(f)

keys = list(cv_dict.keys())

from random import shuffle
shuffle(keys)

L = len(keys)//5
for j in range(5):
    fold_keys = keys[j * size: (j+1) * size]
    fold_dict = {}
    for k in fold_keys:
        fold_dict[k] = cv_dict[k]
    with open(f'cv_fold-{j}.json', 'wt') as f:
        json.dump(fold_dict, f)


# In[ ]:




