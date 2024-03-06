import openai
import time
import re
import random
import json
import os


openai.api_key = 'OPENAI_API_KEY'
def get_ans(gpt_input):

    system_message = '''
    '''
    assistant_message = '''
    '''
    for _ in range(3):
        # time.sleep(5)
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5',
                messages=[{
                    'role': 'system',
                    'content': system_message,
                }, {
                    'role': 'assistant',
                    'content': assistant_message,
                }, {
                    'role': 'user',
                    'content': gpt_input,
                }],
                temperature=0.7,
                max_tokens=1024,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            ans = response['choices'][0]['message']['content']
            return ans
        except Exception as e:
            print('[ERROR]', e)
            ans = '#ERROR#'
            time.sleep(5)
            return ans, 0

with open('/home/dh/pythonProject/DocMSU/data/release/docmsu_all.json', 'r') as f:
    data = json.load(f)
for k,v in data.items():
    if v['is_sar'] == 1:
        ans = get_ans(v['text'])
        print(ans)