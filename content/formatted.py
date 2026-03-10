import json

with open('./content/scraped_dataset.json', encoding='utf-8') as f:
    full_json = json.load(f)

new_data = []

for message in full_json:
    new_data.append({"user": message['author']['id'], "response":message['content']})

new_data.reverse()

with open('./content/formatted.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)