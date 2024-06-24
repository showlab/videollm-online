import json, re

annos = json.load(open('goalstep_livechat_trainval.json'))

new_annos = []
for anno in annos:
    if not anno['conversation']:
        continue
    maintain = True
    anno['duration'] = anno['conversation'][-1]['time'] - anno['conversation'][0]['time']
    if anno['duration'] < 60 or anno['duration'] > 3600:
        continue
    for message in anno['conversation']:
        if 'second' in message['content'] or re.match(r'\b\d+s\b', message['content']): # if the generated content contains time related text, it may leak the future ground-truth 
            maintain = False
            break
    if maintain:
        new_annos.append(anno)