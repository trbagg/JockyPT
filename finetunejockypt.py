## Imports
import gc
import wandb
wandb.login()
import torch
torch.cuda.empty_cache()
print(f"Cuda Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(torch.randn(1).cuda())
gc.collect()
torch.cuda.empty_cache()
import atexit

import json
import transformers
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


model_name = "mistralai/Mistral-7B-Instruct-v0.3"

def main():

	print("Loading models...")

	bnb_config = transformers.BitsAndBytesConfig(
													load_in_4bit=True,
													bnb_4bit_quant_type="nf4",
    												bnb_4bit_use_double_quant=True,
													bnb_4bit_compute_dtype=torch.float16
												)
	
	base_model = transformers.AutoModelForCausalLM.from_pretrained(
																model_name,
																cache_dir="./LLaMA_Quant", 
																quantization_config=bnb_config,
																device_map="auto",
																dtype=torch.float16,
																trust_remote_code=False,
																revision="main",
																use_cache=False,
															).to('cuda')
	
	base_model.config.use_flash_attention = True
	
	print(base_model)
	print(next(base_model.parameters()).dtype)

	tokenizer = transformers.AutoTokenizer.from_pretrained(
																model_name, 
																cache_dir="./LLaMA_Quant",
																add_bos_token=True,
																add_eos_token=False, 
															)
	
	# Mistral is a deconstruct, which reads left during inference, but training sees fixed length sequences, so padding is best set to right for training.
	tokenizer.padding_side = 'right'
	
	print("Padding side set to:", tokenizer.padding_side)
	print(tokenizer.pad_token)

	if tokenizer.pad_token == None:
		tokenizer.pad_token = tokenizer.eos_token #tokenizer.unk_token during inference, eos for consistancy during training
		print("Set padding token to:", tokenizer.pad_token)
	else:
		print("Default padding token set to:", tokenizer.pad_token)

	max_seq_length = 128 # 512 #

	def generate_and_tokenize_prompt(batch):
		all_input_ids = []
		all_attention_masks = []
		all_labels = []

		for conversation in batch['messages']:
			full_text = tokenizer.apply_chat_template(
				conversation,
				tokenize=False,
				add_generation_prompt=False,
				add_special_tokens=True
			)

			tokenized = tokenizer(
				full_text,
				truncation=True,
				max_length=max_seq_length,
				padding="max_length",
				add_special_tokens=False
			)

			input_ids = tokenized["input_ids"]
			labels = [-100] * len(input_ids)

			current_messages = []
			for turn in conversation:
				current_messages.append(turn)
				
				if turn["role"] == "assistant":
					text_up_to_here = tokenizer.apply_chat_template(
						current_messages,
						tokenize=False,
						add_generation_prompt=False,
						add_special_tokens=True
					)
					
					text_up_to_prompt = tokenizer.apply_chat_template(
						current_messages[:-1],
						tokenize=False,
						add_generation_prompt=True,
						add_special_tokens=True
					)
					
					# Tokenize both to find boundaries
					end_idx = len(tokenizer(
						text_up_to_here,
						add_special_tokens=False
					)["input_ids"])
					
					start_idx = len(tokenizer(
						text_up_to_prompt,
						add_special_tokens=False
					)["input_ids"])
					
					# Unmask this assistant's response tokens in labels
					for i in range(start_idx, min(end_idx, len(labels))):
						labels[i] = input_ids[i]

			all_input_ids.append(input_ids)
			all_attention_masks.append(tokenized["attention_mask"])
			all_labels.append(labels)

		return {
			"input_ids": all_input_ids,
			"attention_mask": all_attention_masks,
			"labels": all_labels,
		}

	exhahustive_modules = [
							"q_proj",
							"k_proj",
							"v_proj",
							"o_proj",
							"gate_proj",
							"up_proj",
							"down_proj",
							"lm_head",
							"save_embedding_layers"
						]
	
	recommended_modules = [
							"q_proj",
							"k_proj",
							"v_proj",
							"o_proj",
							"gate_proj",
							"up_proj",
							"down_proj",
						]
	
	default_modules = [
							"q_proj",
							"k_proj",
							"v_proj",
							"o_proj",
						]
	
	minimal_modules = [
							"q_proj",
							"v_proj",
						]

	'''
	# First, freeze all parameters
	for name, param in model.named_parameters():
		param.requires_grad = False

	# Then, unfreeze only the LoRA matrices
	for name, param in model.named_parameters():
		if "lora" in name:
			param.requires_grad = True
	'''

	config = LoraConfig(
							r=16, # lower r for smaller datasets
							lora_alpha=32, # recommended 2x r
							target_modules= default_modules,
							bias="none",
							lora_dropout= 0.05,
							task_type="CAUSAL_LM",
						)

	print(config.target_modules)

	model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True) # *Must* be before get_peft_model
	model = get_peft_model(model, config)
	model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name)


	model.print_trainable_parameters()
	
	model.train()

	### Training Params ###

	dataset_max = -1

	lr = 4.5e-6
	weight_decay = 0.02 # 0.05 for checkpoint overfitting, 0.01 for training
	batch_size = 4
	max_steps = 600
	seed= 335679

	dataset_path = "./content/handmade.json"

	data = load_dataset('json', data_files=dataset_path)
	tokenized_data = data['train'].train_test_split(test_size=0.12, seed=seed)

	

	
	
	tokenized_data['train'] = tokenized_data['train'].map(
						generate_and_tokenize_prompt,
						batched=True,
						batch_size=1,
						remove_columns=tokenized_data["train"].column_names,
					)
	tokenized_data['test'] = tokenized_data['test'].map(
						generate_and_tokenize_prompt, 
						batched=True,
						batch_size=1,
						remove_columns=tokenized_data["test"].column_names,
					)
	lengths = []
	for sample in tokenized_data['train']:
		real_tokens = sum(1 for id in sample['input_ids'] if id != tokenizer.pad_token_id)
		lengths.append(real_tokens)
	
	print(f"Max: {max(lengths)}")
	print(f"Mean: {sum(lengths)/len(lengths):.0f}")
	print(f"95th percentile: {sorted(lengths)[int(len(lengths)*0.95)]}")

	sample = tokenized_data['train'][0]
	non_masked = [l for l in sample['labels'] if l != -100]
	print(tokenizer.decode(non_masked))
	print(tokenizer.decode(sample['input_ids']))

	print()

	
	with open('./train_data.json', 'w', encoding='utf-8') as f:
		tokenized_list = tokenizer.batch_decode(tokenized_data['train']['input_ids'])
		for index, prompt in enumerate(tokenized_list):
			tokenized_list[index] = prompt.replace('<unk>','')
		json.dump(tokenized_list, f, indent=4)
	with open('./test_data.json', 'w', encoding='utf-8') as f:
		tokenized_list = tokenizer.batch_decode(tokenized_data['test']['input_ids'])
		for index, prompt in enumerate(tokenized_list):
			tokenized_list[index] = prompt.replace('<unk>','')
		json.dump(tokenized_list, f, indent=4)
	
	data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

	training_args = transformers.TrainingArguments(
													output_dir = "jockypt-ft",
													run_name=f"jockypt-model:{model_name[:7]}-set:{dataset_max}-seed:{seed}-steps:{max_steps}-batch:{batch_size}-seq:{max_seq_length}-lr:{lr}-decay:{weight_decay}-r:{config.r}-a:{config.lora_alpha}",
													learning_rate=lr,
													per_device_train_batch_size=batch_size,
													per_device_eval_batch_size=batch_size,
													max_steps=max_steps,
													gradient_checkpointing=True,
													weight_decay=weight_decay,
													logging_strategy="steps",
													eval_strategy="steps",
													save_strategy="steps",
													logging_steps=50,
													eval_steps=10,
													save_steps=10,
													load_best_model_at_end=True,
													gradient_accumulation_steps=1,
													warmup_steps= int(max_steps * 0.01),
    												lr_scheduler_type="constant",
													label_names=["labels"],
													bf16=False,
													fp16=True,
													optim="paged_adamw_8bit",
													remove_unused_columns=False,
													dataloader_pin_memory=False,
													report_to="wandb",
												)

	trainer = transformers.Trainer(
									model=model,
									train_dataset=tokenized_data["train"],
									eval_dataset= tokenized_data["test"],
									args=training_args,
									data_collator=data_collator,
								)

	trainer.train(resume_from_checkpoint=False) # False # True , ignore_keys_for_eval=["optimizer", "scheduler"]

	print("Finished training")

def cleanup():
	gc.collect()
	torch.cuda.empty_cache()
	print("Cleanup done at program exit.")

if __name__ == '__main__':
	
	atexit.register(cleanup)
	main()
