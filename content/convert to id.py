import json

with open('./content/handmade.json', encoding='utf-8') as f:
    loaded_json = json.load(f)

conversations = []

for conversation in loaded_json:
    new_data = []
    for message in conversation['messages']:
        new_data.append({"role": message['role'], "content": f'{'[User]' if message['role'] == 'user' else '[Assistant]'} {message['content']}'})
    #new_data.reverse()
    conversations.append({'messages':new_data})

with open('./content/handmade-id.json', 'w', encoding='utf-8') as f:
    json.dump(conversations, f, ensure_ascii=False, indent=4)