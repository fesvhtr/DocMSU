import json
import os

with open('/home/dh/pythonProject/DocMSU/data/release/docmsu_all.json') as f:
    data = json.load(f)
for k,v in data.items():
    print(k)
    print(v)
    break
