#!/usr/bin/env python
# coding: utf-8

# In[1]:


with open('/data/maojingwei/xingchengdan_3code/APR22/all_236.tsv', 'r') as rf:
    lines = rf.readlines()

text_ls = []
for line in lines:
    text_ls.append(''.join(line.split('\t')[1].split(' ')))

import requests, json, ipdb


model_url = "http://30.99.134.178:55660/sentence"

ls = []
for text in text_ls[:10]:
    if len(ls) < 8:
        ls.append(text)
        continue
    # ipdb.set_trace()
    data_sent2model = {"sentences": ls, "task_name": "xingchengdan_3code"}
    print(data_sent2model)
    response = requests.post(model_url, json.dumps(data_sent2model), headers={"Content-type": "application/json"})
    print(response)
    ls = []
if ls:
    data_sent2model = {"sentences": ls, "task_name": "xingchengdan_3code"}
    response = requests.post(model_url, json.dumps(data_sent2model), headers={"Content-type": "application/json"})


# In[ ]:




