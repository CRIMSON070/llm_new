import os
import json
import torch
import gc
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

def main():
    print("Testing GPU access...")
    if not torch.cuda.is_available():
        print("CUDA is not available! Exiting.")
        return
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Initial VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "financebench", "phase9b")
    out_dir = os.path.join(base_dir, "outputs", "phase9b_dpo_pilot")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Audit Dataset Check
    train_path = os.path.join(data_dir, "train_preferences.jsonl")
    if not os.path.exists(train_path):
        print("Dataset not found!")
        return

    # Load and format dataset for DPO
    def load_jsonl(path):
        ds = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # DPO requires plain strings
                ds.append({
                    "prompt": item["prompt"],
                    "chosen": json.dumps(item["chosen"], indent=2),
                    "rejected": json.dumps(item["rejected"], indent=2)
                })
        return ds

    train_data = load_jsonl(train_path)
    train_dataset = Dataset.from_list(train_data)
    
    print(f"Loaded {len(train_dataset)} training examples.")

    # 2. Model Initialization (4-bit NF4)
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_id} in 4-bit...")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto"
        )
        
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        print("Model loaded successfully. Peak VRAM:", torch.cuda.max_memory_allocated(0) / 1024**3, "GB")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. DPO Pilot Training
    print("Initializing DPO Trainer...")
    
    training_args = DPOConfig(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_length=512,
        num_train_epochs=1,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False
    )
    
    try:
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        
        print("Starting pilot training...")
        trainer.train()
        
        print("Saving LoRA adapter...")
        trainer.model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        print("Pilot training complete!")
        
    except Exception as e:
        print(f"OOM or Training Error: {e}")
        print("Peak VRAM at failure:", torch.cuda.max_memory_allocated(0) / 1024**3, "GB")
        
    finally:
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
