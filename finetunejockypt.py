## Imports
import gc
import wandb
wandb.login()
import torch
print(f"Cuda Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(torch.randn(1).cuda())
gc.collect()
torch.cuda.empty_cache()
import atexit

import json
import random
import transformers
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from trl.experimental.orpo import ORPOConfig, ORPOTrainer


model_name = "mistralai/Mistral-7B-Instruct-v0.3" # 

def main():

	print("Loading models...")

	bnb_config = transformers.BitsAndBytesConfig(
													load_in_4bit=True,
													bnb_4bit_quant_type="nf4",
    												bnb_4bit_use_double_quant=True,
													bnb_4bit_compute_dtype=torch.bfloat16
												)
	
	if torch.cuda.get_device_capability()[0] >= 8:
		print("Flash attn 2 available")
	else:
		raise ValueError
	
	base_model = transformers.AutoModelForCausalLM.from_pretrained(
																model_name,
																cache_dir="./LLaMA_Quant", 
																quantization_config=bnb_config,
																device_map="auto",
																dtype=torch.bfloat16,
																trust_remote_code=False,
																revision="main",
																use_cache=False,
																#attn_implementation="flash_attention_2"
															).to('cuda')
	
	base_model.config.use_flash_attention = True
	
	print(base_model)

	

	tokenizer = transformers.AutoTokenizer.from_pretrained(
																model_name, 
																cache_dir="./LLaMA_Quant",
																add_bos_token=True,
																add_eos_token=False, 
															)
	
	print("Padding side set to:", tokenizer.padding_side)
	print("Padding token set to:",tokenizer.pad_token)

	if tokenizer.pad_token == None:
		tokenizer.pad_token = tokenizer.eos_token # tokenizer.unk_token during inference, eos for consistancy during training
		print("Set padding token to:", tokenizer.pad_token)
	else:
		print("Default padding token set to:", tokenizer.pad_token)

	max_seq_length = 128

	exhaustive_modules = [
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


	config = LoraConfig(
							r=16, # lower r for smaller datasets
							lora_alpha=32, # recommended 2x r
							target_modules= recommended_modules,
							bias="none",
							lora_dropout= 0.05,
							task_type="CAUSAL_LM",
						)

	print(config.target_modules)
	
	print(tokenizer.chat_template)

	model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

	### Training Params ###

	dataset_max = -1

	lr = 6e-6
	weight_decay = 0.01 # 0.05 for checkpoint overfitting, 0.01 for training
	batch_size = 1
	max_steps = 600
	beta = 0.50

	dataset_path = "./content/synth_orpo.json" # "./content/handmade_orpo.json" # 

	data = load_dataset('json', data_files=dataset_path)

	print(data)
	print(data["train"][0])
	print(data["train"].column_names)
	print(type(data['train'][0]['chosen']))
	data.shuffle()
	tokenized_data = data['train'].train_test_split(test_size=0.1)


	training_args = ORPOConfig(
													output_dir = "jockypt-ft",
													run_name=f"jockypt-model:{model_name[:7]}-set:{dataset_max}-steps:{max_steps}-batch:{batch_size}-seq:{max_seq_length}-lr:{lr}-decay:{weight_decay}-r:{config.r}-a:{config.lora_alpha}-beta:{beta}",
													learning_rate=lr,
													per_device_train_batch_size=batch_size,
													per_device_eval_batch_size=batch_size,
													max_steps=max_steps,
													gradient_checkpointing=True,
													weight_decay=weight_decay,
													logging_strategy="steps",
													eval_strategy="steps",
													save_strategy="steps",
													max_grad_norm=0.3,
													logging_steps=10,
													eval_steps=10,
													save_steps=10,
													load_best_model_at_end=True,
													gradient_accumulation_steps=4,
													warmup_steps= 30,
    												lr_scheduler_type="constant",
													label_names=["labels"],
													bf16=True,
													fp16=False,
													optim="paged_adamw_8bit",
													remove_unused_columns=False,
													dataloader_pin_memory=False,
													report_to="wandb",
													#ORPO Params
													beta=beta,
													max_length=128,
													max_completion_length=128,
												)

	trainer = ORPOTrainer(
									model=model,
									train_dataset=tokenized_data["train"],
									eval_dataset= tokenized_data["test"],
									args=training_args,
									processing_class=tokenizer,
									peft_config=config
								)
	
	print("chosen logits:", trainer.eval_dataset[0])

	trainer.train(resume_from_checkpoint=True , ignore_keys_for_eval=["optimizer", "scheduler"]) # False #

	print("Finished training")

def cleanup():
	gc.collect()
	torch.cuda.empty_cache()
	print("Cleanup done at program exit.")

if __name__ == '__main__':
	
	atexit.register(cleanup)
	main()
