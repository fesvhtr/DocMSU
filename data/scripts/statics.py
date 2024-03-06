import json

with open('/home/dh/pythonProject/DocMSU/data/release/docmsu_all.json') as f:
    data= json.load(f)
## type: [sar,not_sar]
cnt_dic = {
    'science': [0,0],
    'health': [0,0],
    'sport': [0,0],
    'technology': [0,0],
    'entertainment': [0,0],
    'education': [0,0],
    'business': [0,0],
    'environment': [0,0],
    'politic': [0,0],
}
for k,v in data.items():
    if v['is_sar'] ==1:
        cnt_dic[v['type']][0] += 1
    if v['is_sar'] ==0:
        cnt_dic[v['type']][1] += 1
total = 0
for k,v in cnt_dic.items():
    total += v[0]+v[1]
print(total)
print(cnt_dic)