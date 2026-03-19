import json
import random

def to_orpo(input_path, output_path):

    print(f"Converting {input_path} to {output_path} for ORPO training.")

    with open(input_path, 'r') as f:
        raw_data = json.load(f)
    

    data = []

    for i in range(len(raw_data)):
        data.append({"prompt": raw_data[i]['messages'][:-1], "chosen": [raw_data[i]['messages'][-1]], "rejected": [raw_data[random.randint(0,len(raw_data)-1)]['messages'][-1]]})
        if data[-1][-1]['role'] == 'user':
            pass

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        print("Done.")

if __name__ == '__main__':
    to_orpo('./content/handmade.json', './content/handmade_orpo.json')