
import os
import re
import json
import atexit
import difflib
import discord
import datetime
import requests
from dotenv import load_dotenv
from jocky_pt import cleanup, inference, message_history, local_model_path
from content.automated import usernames, ids, pattern_match

discord_association = {key: value for key, value in zip(usernames,ids)}
ping_pattern = re.compile(r"\[@[A-Za-z0-9_]+\]", re.IGNORECASE)
gif_pattern = re.compile(r"\[gif:[A-Za-z0-9_ ,\-\(\)]+\]", re.IGNORECASE)
dataset_path = "content/ping_info.json"
rolling_messages = {"length": 0, "messages": []}

def tenor_lookup(search_term, lmt = 1):

    api_key = os.getenv('TENOR_API_KEY')

    req = requests.get(
        f"https://tenor.googleapis.com/v2/search?q={search_term}&key={api_key}&limit={lmt}&locale=en_US&media_filter=minimal")

    if req.status_code == 200:
        return json.loads(req.content)
    else:
        return None

def tenor_match(url):

	lmt = 3

	gifs = tenor_lookup(url[1:min(30,len(url))], lmt)
	if(type(gifs) == dict and len(gifs['results']) > 0):
		return gifs['results'][0]['itemurl']

	return ''

def klipy_lookup(search_term, lmt = 1):

	api_key = os.getenv('KLIPY_API_KEY')

	customer_id = 'jocky_api'
	session = requests.Session()
	req = session.get(
						f"https://api.klipy.com/api/v1/{api_key}/gifs/search?q={search_term}&customer_id={customer_id}&per_page={lmt}&locale=en&content_filter=r", 
						headers={'User-Agent': 'Mozilla/5.0'}
					)

	if req.status_code == 200:
		return json.loads(req.content)
	else:
		return None
	
def klipy_match(url):

	lmt = 1

	gifs = klipy_lookup(url[1:min(30,len(url))], lmt)
	if(type(gifs) == dict and gifs['result'] == True and len(gifs['data']['data']) > 0):
		return gifs['data']['data'][0]['file']['hd']['gif']['url']

	return ''

def giphy_lookup(search_term, lmt = 1):

	api_key = os.getenv('GIPHY_API_KEY')

	customer_id = 'jocky_api'
	session = requests.Session()
	req = session.get(
						f"https://api.giphy.com/v1/gifs/search?api_key={api_key}&q={search_term}&random_id={customer_id}&limit={lmt}&lang=en&rating=r", 
						headers={'User-Agent': 'Mozilla/5.0'}
					)

	if req.status_code == 200:
		return json.loads(req.content)
	else:
		return None
	
def giphy_match(url):

	lmt = 1

	gifs = giphy_lookup(url[1:min(30,len(url))], lmt)
	if(type(gifs) == dict and len(gifs['data']) > 0):
		ret = gifs['data']
		ret = ret[0]
		ret = ret['embed_url']
		return gifs['data'][0]['embed_url']

	return ''

def ping_match(ping, discord_association):
	
	best_val = 0
	chosen = None

	for username in usernames:
		val = difflib.SequenceMatcher(None,ping, username).ratio()
		if val > best_val:
			best_val = val
			chosen = discord_association.get(ping)

	return f"<@{chosen if chosen is not None else f'(A ping error happened when attempting to ping: {ping} as {chosen})'}>" #f"<ping:@{association.get(chosen, "Key not found")}>"

def unsanitize(response):
	match = True
	while match:
		match = ping_pattern.search(response)
		if match:
			response = response[:match.start()] + ping_match(response[match.start()+2:match.end()-1], discord_association) + response[match.end():]

	match = True
	while match:
		match = gif_pattern.search(response)
		if match:
			gif_link = giphy_match(response[match.start()+5:match.end()-1])
			response = f'{response[:match.start()]} {gif_link} {response[match.end():]}'
	
	return response

def bot_main():

	load_dotenv()
	TOKEN = os.getenv('DISCORD_TOKEN')
	GUILD = os.getenv('DISCORD_GUILD')

	class CustomClient(discord.Client):
		async def on_ready(self):
			guild = discord.utils.get(self.guilds, name=GUILD)
			print(f'{self.user} has connected to Discord!')
			print(f'Registered to {guild.name}: {guild.id}')
			channel = self.get_channel(1361461684094373898)
			if channel is not None:
				await channel.send(f'Loading checkpoint: {local_model_path}', reference=None, mention_author=False)
			else:
				raise ValueError("Failed to load Test Server Channel")

		async def on_message(self, message):
			global rolling_messages

			if msg := check_commands(message=message.content, history=rolling_messages['messages']):
				await message.channel.send(msg[1], reference=message, mention_author=False)
				if msg[0] == True:
					await client.close()
				return
			
			if message.author == client.user:
				return
			
			temperature = 0.7

			if match := re.search(r'temperature=[0-9]*\.[0-9]+', message.content):
				temperature = message.content[match.start()+len('temperature='):match.end()]
				message.content = message.content[:match.start()] + message.content[match.end():]

			

			inference_message, _ = pattern_match(message.content) # sanitize gifs, emotes, etc. to match dataset pattern
			message_response = inference(message=inference_message, history=rolling_messages['messages'] if not isinstance(message.channel, discord.DMChannel) else [], member=message.author, temperature=float(temperature))
			print("Original message:", message_response)
			message_history(rolling_messages, user='user', msg=inference_message) # save sanitized output (for input consistancy)
			message_history(rolling_messages, user='assistant', msg=message_response)
			message_response = unsanitize(message_response) # then after saving, apply discord formatting

			if message_response == '':
				message_response = '>thinking...'
			await message.channel.send(message_response, reference=message, mention_author=False)
			print()

		async def on_error(self, event, *args, **kwargs):
			with open('err.log', 'a') as f:
				if event == 'on_message':
					f.write(f'Unhandled message: {args[0]}\n')
				else:
					raise
	intents=discord.Intents.all()
	intents.presences = False
	client = CustomClient(intents=intents)
	client.run(TOKEN)
	
def check_commands(message, history):
	if message[:5].lower() == '!save':
		date = datetime.datetime.now()
		output_file = f"./conversations/{date.month}.{date.day}.{date.year}_{date.hour}-{date.minute}-{date.second}.json"
		os.makedirs(os.path.dirname(output_file), exist_ok=True)
		with open(output_file, 'w', encoding='utf-8') as file:
			conversation=[]
			for entry in history:
				conversation.append(entry)
			json.dump({'messages': conversation}, file, indent=4)
			return (False, f"Saved to: {output_file}")
	elif message[:7].lower() == '!ignore':
		return (False, f'Ignored {message[:20]}...')
	elif message[:5].lower() == '!dump':
		history.clear()
		return (False, 'Dumped memory.')
	elif message[:9].lower() == '!shutdown':
		return (True, 'Shutting down.')


if __name__ == '__main__':

	atexit.register(cleanup)
	bot_main()