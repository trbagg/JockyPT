import gc
import json
import torch
import random
gc.collect()
torch.cuda.empty_cache()
print(torch.cuda.is_available())
print(torch.randn(1).cuda())
torch.cuda.empty_cache()
import transformers


def main(input_path, output_path, iters = 3):
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    bnb_config = transformers.BitsAndBytesConfig(
                                                load_in_4bit=True,
                                                bnb_4bit_use_double_quant=True,
                                                bnb_4bit_quant_type="nf4",
                                                bnb_4bit_compute_dtype=torch.bfloat16
                                            )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, cache_dir="./LLaMA_Quant")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
                                                    model_id,
                                                    cache_dir="./LLaMA_Quant", 
                                                    quantization_config=bnb_config,
                                                    device_map="auto",
                                                    dtype=torch.bfloat16,
                                                    trust_remote_code=False,
                                                    revision="main",
                                                    use_cache=False,
                                                )
    model.config.use_flash_attention = True

    fluff(input_path=input_path, output_path=output_path, iters=iters, tokenizer=tokenizer, model=model)

def fluff(input_path, output_path, iters, tokenizer, model):

    print(f"Fluffing {input_path} to {output_path}.")

    with open(input_path, 'r') as f:
        raw_data = json.load(f)

    data = []

    for conversation in raw_data:
        data.append(conversation)
        prompt = conversation['prompt']
        chosen = conversation['chosen']
        rejected = conversation['rejected']
        instruction = [
            {
                "role": "user",
                "content": "Rephrase the following sentence to have a different wording but exactly the same meaning and intent." \
                "Do not add any extra information or explanation. Maintain the same manner of speaking, including misspelling, any lack of punctuation, tone, mannerisms, crudeness, etc."\
                f"You will be provided the conversation for context to shape the tone and manner in how you respond, the original response to reword, and a rejected example, which should be used as guidance for what *NOT* to respond with. \n\nContext: {'\n'.join([str(s['content']) for s in prompt])} Original: {chosen[0]['content']} Rejected: {rejected[0]['content']} Reworded:"
            },
        ]

        formatted_instruction = tokenizer.apply_chat_template(
            instruction, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        batch_to_generate = [formatted_instruction] * iters
        
        inputs = tokenizer(batch_to_generate, return_tensors="pt", padding=True).to(model.device)

        input_ids = inputs["input_ids"].to('cuda')

        generated_outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )

        decoded_texts = tokenizer.batch_decode([text[input_ids.shape[1]:] for text in generated_outputs], skip_special_tokens=True) # [generated_outputs[0][input_ids.shape[1]:]]
        
        for output_text in decoded_texts:
            data.append({'prompt': prompt, 'chosen': {'role': 'assistant', 'content': output_text}, 'rejected': rejected})


    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        print("Done.")

if __name__ == '__main__':
    main('./content/handmade_orpo.json', './content/fluff_orpo.json')