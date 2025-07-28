import builtins
_original_open = builtins.open
def _utf8_open(file, mode='r', *args, **kwargs):
    if 'b' not in mode:
        kwargs.setdefault('encoding', 'utf-8')
    return _original_open(file, mode, *args, **kwargs)
builtins.open = _utf8_open

import unsloth
import glob
import json
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import sys

# Ensure UTF-8 printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def to_sft(example):
    """Convert conversation to prompt/completion pairs"""
    conv = example["conversations"]
    
    human_idxs = [i for i, msg in enumerate(conv) if msg["from"] == "human"]
    if not human_idxs:
        return {"prompt": None, "completion": None}
    
    training_pairs = []
    
    for human_idx in human_idxs:
        if human_idx + 1 < len(conv) and conv[human_idx + 1]["from"] == "gpt":
            prompt = ""
            for msg in conv[:human_idx]:
                role = "user" if msg["from"] == "human" else "assistant"
                prompt += f"<|im_start|>{role}\n{msg['value']}<|im_end|>\n"
            
            prompt += f"<|im_start|>user\n{conv[human_idx]['value']}<|im_end|>\n<|im_start|>assistant\n"
            
            completion = conv[human_idx + 1]['value'] + "<|im_end|>"
            
            training_pairs.append({"prompt": prompt, "completion": completion})
    
    if training_pairs:
        return training_pairs[-1]
    else:
        return {"prompt": None, "completion": None}

def formatting_func(example):
    """Format prompt and completion for training"""
    text = example["prompt"] + example["completion"]
    
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        return_attention_mask=True,
    )
    
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    prompt_tokenized = tokenizer(
        example["prompt"],
        add_special_tokens=False,
        truncation=True,
    )["input_ids"]
    
    labels = input_ids.copy()
    
    prompt_end_idx = None
    
    for i in range(len(input_ids) - len(prompt_tokenized) + 1):
        if input_ids[i:i+len(prompt_tokenized)] == prompt_tokenized:
            prompt_end_idx = i + len(prompt_tokenized)
            break
    
    if prompt_end_idx is None:
        prompt_with_special = tokenizer(
            example["prompt"],
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        prompt_end_idx = min(len(prompt_with_special), len(input_ids) - 1)
        prompt_end_idx = max(prompt_end_idx, 0)
    
    for i in range(min(prompt_end_idx, len(labels))):
        labels[i] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    
if __name__ == "__main__":
    # Gather all individual JSON files
    json_files = glob.glob("data/preprocessed/*.json")
    print(f"Found {len(json_files)} JSON files")

    # Load them into a single Hugging Face DatasetDict
    raw = load_dataset(
        "json",
        data_files=json_files,                                   
    )

    # Split into train/validation (90/10)
    splits = raw["train"].train_test_split(test_size=0.1, seed=2025)

    # Convert to SFT format
    sft_mapped = splits.map(
        to_sft,
        remove_columns=["conversations"],
        batched=False,
    )

    # Filter out None results
    sft_ds = sft_mapped.filter(lambda x: x["prompt"] is not None and x["completion"] is not None)

    print(f"Training examples: {len(sft_ds['train'])}")
    print(f"Validation examples: {len(sft_ds['test'])}")

    # Show a sample to verify
    if len(sft_ds['train']) > 0:
        sample = sft_ds['train'][0]
        print("\nSample training example:")
        print(f"Prompt: {sample['prompt'][:200]}...")
        print(f"Completion: {sample['completion'][:100]}...")

    # Load & quantize the base model
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        max_seq_length=2048
    )

    # Add QLoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj","k_proj","v_proj","up_proj","down_proj","o_proj"],
        use_rslora=True,
        use_gradient_checkpointing="unsloth"
    )

    # Choose ChatML template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
    )
    
    tokenizer.truncation_side = "left"

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir="whatsapp-sft",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=200,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=10,
        eval_steps=10,
        do_eval=True,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        eval_strategy="steps",
        save_strategy="steps",
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        dataloader_num_workers=0,
    )

    # Format datasets
    print("\nFormatting training data...")
    formatted_train = sft_ds["train"].map(
        formatting_func,
        remove_columns=["prompt", "completion"],
        batched=False,
        num_proc=1,
    )

    print("Formatting validation data...")
    formatted_test = sft_ds["test"].map(
        formatting_func,
        remove_columns=["prompt", "completion"],
        batched=False,
        num_proc=1,
    )
    
    # Sanity‑check for fully masked examples
    print("Checking for any fully‑masked eval examples…")
    for i, ex in enumerate(formatted_test):
        if all(label == -100 for label in ex["labels"]):
            print(f" ⚠️ Example {i} has all labels = -100 (will produce NaN loss)")
            break
    else:
        print(" ✅ No fully‑masked eval examples found.")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_train,
        eval_dataset=formatted_test,
        args=training_args,
        tokenizer=tokenizer,
        packing=False,
        dataset_text_field=None,
        max_seq_length=2048,
    )

    # Start fine-tuning
    print("\nStarting training...")
    print("This will take a while. Watch the loss decrease!")
    print("-" * 50)
    
    trainer.train()

    print(f"\nBest model checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Best metric value: {trainer.state.best_metric}")

    # Save the LoRA adapters
    print("\nSaving model...")
    model.save_pretrained("whatsapp-lora-adapter")
    tokenizer.save_pretrained("whatsapp-lora-adapter")

    print("\nTraining complete! Model saved to whatsapp-lora-adapter")

    # Test the model
    print("\nQuick test of the trained model:")
    FastLanguageModel.for_inference(model)

    test_prompts = ["Hello!", "How are you?", "What's up?"]
    
    for prompt in test_prompts:
        print(f"\nUser: {prompt}")
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=50,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        print(f"Assistant: {response}")

    print("\n" + "="*50)
    print("Training complete! Use the improved chat script to talk with your model.")
    print("Model saved to: whatsapp-lora-adapter")