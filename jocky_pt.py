import gc
import torch
from finetunejockypt import model_name
gc.collect()
torch.cuda.empty_cache()
print(torch.cuda.is_available())
print(torch.randn(1).cuda())
torch.cuda.empty_cache()
import transformers
from peft import PeftModel

local_model_path = "./jockypt-pt/checkpoint-430"

print("Loading models...")

cache_dir="./LLaMA_Quant"
sys_instruct_path = "./content/system_instruction.txt"

with open(sys_instruct_path, 'r') as f:
	sys_instruct = f.read()

bnb_config = transformers.BitsAndBytesConfig(
												load_in_4bit=True,
												bnb_4bit_use_double_quant=True,
												bnb_4bit_quant_type="nf4",
												bnb_4bit_compute_dtype=torch.float16
											)

base_model = transformers.AutoModelForCausalLM.from_pretrained(
	model_name,
	cache_dir=cache_dir, 
	quantization_config=bnb_config,
	dtype=torch.float16,
	device_map="auto",
	trust_remote_code=True,
)

base_model.config.use_flash_attention = True

eval_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, 
															add_bos_token=True,
															add_eos_token=False,  
															trust_remote_code=True)

peft_model = PeftModel.from_pretrained(
										base_model, 
									 	local_model_path,
										device_map="auto",
										dtype=torch.float16
									)



def inference(message, history, no_bot = False, member = 'Yogipanda', temperature = None):
	print(f"Infering {member}'s message: {message}")
	try:
		if no_bot:
			message = input("User: ")
		prompt_input = eval_tokenizer.apply_chat_template(
															#[{"role": "system", "content": f"{sys_instruct}"}] + # mistral does not support sys prompts explicitly, must be trained on it
															[{"role": entry['role'], "content": entry['content']} for entry in history] + #
															[{"role": "user", "content": message}],
															tokenize=False,
															add_generation_prompt=True,
															add_special_tokens=False,
														)
		inputs = eval_tokenizer(prompt_input, return_tensors='pt').to('cuda')

		input_ids = inputs["input_ids"].to('cuda')
		
		peft_model.eval()

		
		with torch.no_grad():
			output = base_model.generate(
											input_ids, 
											attention_mask=inputs["attention_mask"].to('cuda'), 
											eos_token_id=eval_tokenizer.eos_token_id,
											pad_token_id=eval_tokenizer.pad_token_id,
											max_new_tokens=128, # 512
											no_repeat_ngram_size=3,
											temperature= 0.7 if temperature is None else temperature, # Note: this is not where initial temp is set
											repetition_penalty=1.3,
											top_p=0.85,
											top_k=30,
											do_sample=True,
										)
			
			print(output)
			generated_text = eval_tokenizer.batch_decode([output[0][input_ids.shape[1]:]], skip_special_tokens=True)

		return generated_text[0]

	except KeyboardInterrupt:
		return





def message_history(history, user, msg, context_length = 8192, max_messages = 16):
	history['messages'].append({"role": user, "content": msg})
	tokenized_history = tokenize_history(history=history['messages'])
	history['length'] = len(eval_tokenizer.tokenize(tokenized_history))
	while history['length'] >= context_length:
		history['messages'].pop(0)
		tokenized_history = tokenize_history(history=history['messages'])
		history['length'] = len(eval_tokenizer.tokenize(tokenize_history(history=history['messages'])))
	
	while max_messages > -1 and len(history['messages']) > max_messages:
		history['messages'].pop(0)
		if history['messages'][0]['role'] != 'user':
			history['messages'].pop(0)
		history['length'] = len(eval_tokenizer.tokenize(tokenize_history(history=history['messages'])))

def tokenize_history(history):
	return eval_tokenizer.apply_chat_template(
												history,
												tokenize=False,
												add_generation_prompt=True,
												add_special_tokens=False,
											)

def cleanup():
	gc.collect()
	torch.cuda.empty_cache()
	print("Cleanup done at program exit.")

if __name__ == '__main__':

	cleanup()
	print("This file is not meant to be run. Perhaps you meant to run jocky_bot.py?")
	exit(1)