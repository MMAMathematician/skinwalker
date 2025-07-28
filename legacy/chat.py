import builtins
_original_open = builtins.open

def _utf8_open(file, mode='r', *args, **kwargs):
    if 'b' not in mode:
        kwargs.setdefault('encoding', 'utf-8')
    return _original_open(file, mode, *args, **kwargs)

builtins.open = _utf8_open

import sys
import os
from datetime import datetime
import torch
import json
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from peft import PeftModel
from transformers import StoppingCriteria, StoppingCriteriaList

# Ensure UTF-8 printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class RepetitionStoppingCriteria(StoppingCriteria):
    """Stop generation if the same token repeats too many times"""
    def __init__(self, tokenizer, max_repetitions=3):
        self.tokenizer = tokenizer
        self.max_repetitions = max_repetitions
    
    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids[0]) < self.max_repetitions:
            return False
        last_tokens = input_ids[0][-self.max_repetitions:].tolist()
        if len(set(last_tokens)) == 1:
            return True
        text = self.tokenizer.decode(input_ids[0][-20:], skip_special_tokens=True)
        words = text.split()
        if len(words) >= 6 and words[-3:] == words[-6:-3]:
            return True
        return False


def load_finetuned_model():
    print("Loading fine-tuned model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        load_in_4bit=True,
        device_map="auto",
        max_seq_length=2048
    )
    model = PeftModel.from_pretrained(model, "whatsapp-lora-adapter")
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    print("Model loaded successfully!")
    return model, tokenizer


def load_base_model():
    print("Loading base Llama model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        load_in_4bit=True,
        device_map="auto",
        max_seq_length=2048
    )
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    print("Base model loaded successfully!")
    return model, tokenizer


def clean_response(response):
    words = response.split()
    if len(words) > 3:
        for i in range(1, min(5, len(words)//2)):
            if words[-i:] == words[-2*i:-i]:
                response = ' '.join(words[:-i])
                break
    sentences = response.split('.')
    if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
        response = '.'.join(sentences[:-1]) + '.'
    response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    if response and response[-1] not in '.!?':
        last_word = response.split()[-1] if response.split() else ""
        if len(last_word) < 3 and last_word.lower() not in ['i', 'a', 'am', 'is', 'it', 'at', 'to', 'do', 'go', 'no', 'so']:
            response = ' '.join(response.split()[:-1])
    return response


def generate_response(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7, top_p=0.8, top_k=30):
    system_prompt = open('system-prompt.txt').read()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048, padding=True).to(model.device)
    stopping_criteria = StoppingCriteriaList([RepetitionStoppingCriteria(tokenizer, max_repetitions=3)])
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            temperature=0.7,
            top_p=0.8,
            top_k=30,
            do_sample=True,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
            length_penalty=0.8,
            no_repeat_ngram_size=3,
            early_stopping=True,
            stopping_criteria=stopping_criteria,
            use_cache=True
        )
    gen_ids = outputs[0][len(inputs.input_ids[0]):]
    resp = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return clean_response(resp) or "I understand. How can I help you?"


def chat_interface(use_finetuned=True):
    if use_finetuned:
        model, tokenizer = load_finetuned_model()
        model_type = "WhatsApp-style fine-tuned"
    else:
        model, tokenizer = load_base_model()
        model_type = "Base Llama"

    system_prompt = open('system-prompt.txt', encoding='utf-8').read()

    settings = {"temperature": 0.1, "max_tokens": 150, "top_p": 0.6, "top_k": 20}

    conversation = []
    if not os.path.exists("convos"):
        os.makedirs("convos")

    print("\n" + "="*50)
    print(f"Chat with {model_type} model!")
    print("Type 'quit', 'exit', or 'bye' to end")
    print("Type 'settings' to adjust generation parameters")
    print("="*50 + "\n")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nGoodbye! 👋")
            break

        if user_input.lower() == 'settings':
            print("\nCurrent settings:")
            for k, v in settings.items():
                print(f"  {k}: {v}")
            print("\nTo change: settings <key> <value>")
            continue

        if user_input.lower().startswith('settings '):
            parts = user_input.split()
            if len(parts) == 3 and parts[1] in settings:
                try:
                    settings[parts[1]] = float(parts[2]) if parts[1] != 'max_tokens' else int(parts[2])
                    print(f"Updated {parts[1]} to {settings[parts[1]]}")
                except ValueError:
                    print("Invalid value")
            else:
                print("Usage: settings <key> <value>")
            continue

        conversation.append({"role": "User", "content": user_input})

        print("\nBot: ", end="", flush=True)
        response = generate_response(model, tokenizer, user_input, max_new_tokens=settings['max_tokens'], temperature=settings['temperature'], top_p=settings["top_p"], top_k=settings['top_k'])
        print(response)

        conversation.append({"role": "Bot", "content": response})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("convos", f"{timestamp}.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_type}\n")
        f.write("System prompt:\n" + system_prompt + "\n\n")
        f.write("Parameters:\n")
        f.write(f"temperature: {settings['temperature']}\n")
        f.write(f"top_p: {settings['top_p']}\n")
        f.write(f"top_k: {settings['top_k']}\n\n")
        f.write("Conversation:\n")
        for msg in conversation:
            f.write(f"{msg['role']}: {msg['content']}\n")
    print(f"Conversation saved to {log_file}")


if __name__ == "__main__":
    print("\nLlama Model Chat Interface")
    print("="*50)
    print("1. Chat with fine-tuned WhatsApp-style model")
    print("2. Chat with base Llama model")
    choice = input("\nChoice (1-2): ").strip()
    if choice == "1":
        chat_interface(use_finetuned=True)
    elif choice == "2":
        chat_interface(use_finetuned=False)
    else:
        print("Invalid choice. Please run the script again and select 1 or 2.")
