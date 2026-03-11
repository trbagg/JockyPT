import os
import re
import json
import ijson
import random
import dismoji
import requests
from dotenv import load_dotenv
from collections import defaultdict

total_gifs = 0

ping_info_path = "./content/ping_info.json"

with open(ping_info_path, 'r') as f:
    ping_info_data = json.load(f)

usernames = []
ids = []

for key, value in ping_info_data.items():
    usernames.append(key)
    ids.append(value)

emoticon_pattern = re.compile(
        "[\U0001F600-\U0001F64F"    # Emoticons
        "\U0001F300-\U0001F5FF"     # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"     # Transport & Map Symbols
        "\U0001F700-\U0001F77F"     # Alchemical Symbols
        "\U0001F780-\U0001F7FF"     # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"     # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"     # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"     # Chess Symbols
        "\U0001FA70-\U0001FAFF"     # Symbols and Pictographs Extended-A
        "\U00002700-\U000027BF"     # Dingbats
        "\U00002600-\U000026FF"     # Miscellaneous Symbols (includes ♥️)
        "]+",
        flags=re.UNICODE
    )


load_dotenv()
API_KEY = os.getenv('TENOR_API_KEY')

def tenor_tags(link):
    return_phrase = "[gif:"

    session = requests.Session()
    match = re.search(r'(\d{5,})(?!.*\d)', link)
    if match is None:
        return 'Error fetching Tenor API'
    req = session.get(
                            url=f"https://tenor.googleapis.com/v2/posts?ids={match.group(0)}&key={API_KEY}&limit={1}media_filter=minimal", 
                            headers={'User-Agent': 'Mozilla/5.0'}
                        )
    if req.status_code != 200:
        return None
    
    response = json.loads(req.content)
    if len(response['results']) > 0:
        for tag in response['results'][0]['tags'][:6]:
            return_phrase += f' {tag},'
        return return_phrase[:-1] + ']'
    else:
        return '[gif:]'

def tenor_keywords(link):
    keywords = f"[gif: {link[len('https://tenor.com/view/'):]}]"
    keywords = re.sub(r'[0-9][0-9]+', '', keywords)
    keywords = re.sub(r'-+', ' ', keywords)
    if keywords[-2] == ' ':
        keywords= keywords[:-2] + keywords[-1:]
    keywords = keywords.split()
    for index, word in enumerate(keywords.copy()):
        if re.search(r'%[A-Z0-9]', word):
            keywords.remove(word)
        elif 'gif' in word and index > 0:
            keywords.remove(word)
        elif 'giphy' in word:
            keywords.remove(word)
    seen = set()
    list = keywords.copy()
    
    keywords = [x for x in list if not (x in seen or seen.add(x))]
    
    if ']' not in keywords[-1]:
         keywords[-1] =  keywords[-1] + ']'
    keywords = ' '.join(keywords)
    if len(keywords) <= 8:
        return None
    else:
        return keywords

def format_json(entries = -1, use_multi_turn = True, additive_dataset = False, owner_messages_only = False):

    pings = defaultdict(int)

    def single_turn(f):

        successful_conversions = 0
        attempted_conversions = 0

        current_percentage = -1
        single_data = []
        
        dump_data = []
        stored_json_message = None
        for json_message in ijson.items(f,"item"):
            attempted_conversions += 1
            if entries > -1 and successful_conversions > entries :
                break
            

            if stored_json_message == None:
                if json_message['user'] != '162321857134067713' and owner_messages_only == True:
                    continue
                stored_json_message = json_message
                continue
            
            if current_percentage < int(successful_conversions/entries * 100):
                current_percentage = int(successful_conversions/entries * 100)
                print(f'{current_percentage}%')

            if successful_conversions == 56:
                pass

            prev_message = sanitize(
                                        stored_json_message, pings=pings, 
                                        emoticon_pattern=emoticon_pattern,
                                    )
            message = sanitize(
                                json_message, pings=pings, 
                                emoticon_pattern=emoticon_pattern,
                            )

            if type(message) is tuple or type(prev_message) is tuple:
                if type(prev_message) is tuple:
                    stored_json_message = None
                dump_data.append({"message": message, "response": prev_message})
                continue
            else:
                successful_conversions += 2
            

            global total_gifs

            if '[gif:' in prev_message:
                total_gifs += 1
            if '[gif:' in message:
                total_gifs += 1

            stored_json_message = None



            single_data.append({"messages": [{"role": "user", "content": prev_message}, {"role": "assistant", "content": message}]})
        return successful_conversions,attempted_conversions,current_percentage, single_data, dump_data
    def multi_turn(f):
            
            successful_conversions = 0
            attempted_conversions = 0

            multi_turn = []

            participants = (None, None)
            user_turn = True

            conversation = []
            for json_message in ijson.items(f,"item"):
                attempted_conversions += 1
                if entries > -1 and successful_conversions > entries:
                    break
                
                message = sanitize(
                                    json_message, pings=pings, 
                                    emoticon_pattern=emoticon_pattern,
                                )

                if type(message) is tuple:
                    continue

                if json_message['user'] not in participants:
                    if participants[0] == None:
                        if owner_messages_only == False or json_message['user'] != '162321857134067713':
                            participants = (json_message['user'], None)
                        else:
                            continue
                    elif participants[1] == None:
                        if owner_messages_only == False or json_message['user'] == '162321857134067713':
                            participants = (participants[0], json_message['user'])
                        else:
                            continue
                    elif owner_messages_only == False or (participants[0] != '162321857134067713' and participants[1] == '162321857134067713'):
                        if conversation[-1]['role'] == 'user':
                            conversation = conversation[:-1]
                        multi_turn.append({"messages": conversation})
                        if conversation[0]['role'] == 'assistant':
                            raise SyntaxError
                        conversation = []
                        participants = (json_message['user'], None)
                        user_turn = True
                    else:
                        continue

                if json_message['user'] == participants[0]:
                    if user_turn == False:
                        conversation.append({"role": "assistant", "content": ""})
                    conversation.append({"role": "user", "content": message})
                    user_turn = False
                elif json_message['user'] == participants[1]:
                    if user_turn == False:
                        conversation.append({"role": "assistant", "content": message})
                    else:
                        conversation[-1]['content'] += ' ' + message # '\n' would be more accurate
                    user_turn = True
                else:
                    raise IndexError


                if '[gif:' in message:
                    global total_gifs
                    total_gifs += 1
                
                successful_conversions += 1

                # ijson streams json, so the entire length cannot be determined for a percentage report.
                if successful_conversions % 1000 == 0:
                    print(f'{successful_conversions} succesful conversions') 
            
            return successful_conversions,attempted_conversions, multi_turn
    
    with open('./content/formatted.json', "rb") as f:
        if use_multi_turn:
            successful_conversions,attempted_conversions, multi_data = multi_turn(f=f)
        elif not use_multi_turn or additive_dataset:
            successful_conversions,attempted_conversions, single_data = single_turn(f=f)
    if successful_conversions < attempted_conversions:
        print(f"Culled {attempted_conversions-successful_conversions} entries out of {attempted_conversions}")

    with open(f'./content/final_output.json', 'w', encoding='utf-8') as f:
        if multi_data is not None:
            if additive_dataset:
                single_data += multi_data
            else:
                single_data = multi_data
            random.shuffle(single_data)
        json.dump(single_data, f, ensure_ascii=False, indent=4)
        print("Dumped to final_output")
    
    print(f'Gif ratio: {int(total_gifs/successful_conversions*100)}% ({total_gifs}/{successful_conversions})')

def sanitize(message, emoticon_pattern, pings):

    if not message:
        return (message['response'], "No message content")

    message_only = message['response']

    message['response'], message_only = pattern_match(message=message['response'], pings=pings)

    if emoticon_pattern.search(message['response']):
        return (message['response'], "Message contains emoticon")
    
    message['response'] = message['response'].replace('’','')
    
    message['response'] = message['response'].replace('\"','')
    
    message['response'] = message['response'].replace('\'','')

    message['response'] = message['response'].replace('|','')

    message['response'] = message['response'].replace('`','')

    if any(sub in message['response'].lower() for sub in ['youtu.be', 'https://', '.com', 'services.flhsmv.gov', 'itch.io', 'www.twitch.tv', 'media.discordapp.net', 'www.vintagestory.at']) :
        return (message['response'], "Message contains link")

    if message['response'].isdigit():
        return (message['response'], "Message is number")

    meaningful_length = message['response'].replace(' ', '')
    if len(meaningful_length) <= 3:
        return (message['response'], "Message length <= 3")

    if '\n' in message['response']:
        return (message['response'], "Message contains newline formatting")
    
    if '-play' in message['response']:
        return (message['response'], "Message contains play command")

    if any(special_token in message['response'] for special_token in ['###','<s>','</s>', 'Instruction:', 'Response:']):
        return (message['response'], "Message contains special tokens")
    
    message['response'] = message['response'].replace('#','')
    
    if re.search(r'[0-9][0-9][0-9]+', message['response']): # numbers two or more digits long
        return (message['response'], "Message contains number with three or more sequential digits")
    
    if match := re.search(r'(.)\1{3,}+', message['response']): # strings with the same character five or more times in a row
        message['response'] = message['response'][:match.start()] + (message['response'][match.start()+1] * 3) + message['response'][match.end():]
        #return None

    if match := re.search(r'\[@[A-Za-z]+\]', message['response']):
        if len(message['response']) - len(message['response'][match.start():match.end()]) <= 2: # throwout ping-to-join messages 
            return (message['response'], "Message contains only a ping")

    while match := re.search(r'\.\.+', message['response']):
        message['response'] = message['response'][:match.start()] + message['response'][match.end():]

    while match := re.search(r'%[A-Z0-9]', message['response']): #a-z
        return (message['response'], "Message contains hexidecimal")

    message['response'] = message['response'].replace('%','') # has to be after hexidecimal matching to cleanup

    while '  ' in message['response']:
        message['response'] = message['response'].replace('  ', ' ').strip()

    if message['response'] == 'None':
        return (message['response'], "Message is None type")
    
    if len(message_only.replace(' ', '')) <= 3:
        return (message['response'], "Message meaningful length <= 3")

    return message['response'].strip()
    
def pattern_match(message, pings = None):
    discord_association = {key: value for key, value in zip(
        ids,usernames)}

    ping_pattern = re.compile(r"<@[0-9]+>", re.IGNORECASE)

    emoji_pattern = re.compile(r"<a*:[A-Za-z0-9_]+:[0-9]+>", re.IGNORECASE)

    emoji_word_pattern = re.compile(r"<a*:[A-Za-z0-9_]+:", re.IGNORECASE)

    tenor_pattern = re.compile(r"https://tenor\.com/", re.IGNORECASE)
    
    message = dismoji.demojize(message)

    message_only = message
    
    while True:
        match = ping_pattern.search(message)
        if not match:
            break

        message_pre = message[:match.start()]
        id_to_username = discord_association.get(message[match.start()+2:match.end()-1], '')
        id_to_username = (message[match.start()+2:match.end()-1]) if id_to_username == '' else id_to_username
        if pings:
            pings[id_to_username] += 1

        message_replacement = f'<@{id_to_username}>'
        message_post = message[match.end():]
        message_only.replace(message_replacement, '')
        message = f"{message_pre}{message_replacement if message_replacement != '<@>' else ''}{message_post}"
    
    while True:
        match = emoji_pattern.search(message)
        if not match:
            break
        emoji_word_match = emoji_word_pattern.search(message)
        message_pre = message[:match.start()]
        message_replacement = f"{message[emoji_word_match.start()+1:emoji_word_match.end()]}"
        message_post = message[match.end():]
        message = f"{message_pre}{message_replacement}{message_post}"

    while True:
        match = tenor_pattern.search(message)
        if not match:
            break
        link_end = message[match.start():].find(' ')
        if link_end == -1 or link_end == len(message):
            link_end = len(message)
        message_pre = message[:match.start()]
        message_replacement = tenor_tags(message[match.start():link_end])
        if message_replacement == 'None':
            return (message, "Message contains invalid tenor gif")
        message_post = message[link_end:]
        message = f"{message_pre}{message_replacement}{message_post}"
    
    return message, message_only

if __name__ == "__main__":
    format_json()