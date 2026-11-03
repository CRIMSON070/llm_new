import os
import json
import time
import torch
import gc
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
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
    out_dir = os.path.join(base_dir, "outputs", "phase9b_dpo_phase2")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Dataset Loading
    train_path = os.path.join(data_dir, "train_preferences.jsonl")
    val_path = os.path.join(data_dir, "validation_preferences.jsonl")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Dataset not found!")
        return

    def load_jsonl(path):
        ds = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                ds.append({
                    "prompt": item["prompt"],
                    "chosen": json.dumps(item["chosen"], indent=2),
                    "rejected": json.dumps(item["rejected"], indent=2)
                })
        return ds

    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path)
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    print(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples.")

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
        
        print("Model loaded successfully. Base model VRAM:", torch.cuda.max_memory_allocated(0) / 1024**3, "GB")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Controlled DPO Phase 2 Configuration (3 Epochs)
    print("Initializing DPO Trainer for 3 epochs...")
    
    training_args = DPOConfig(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_length=512,
        num_train_epochs=3,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        eval_strategy="epoch",
        per_device_eval_batch_size=1,
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False
    )
    
    config_dict = {
        "model_id": model_id,
        "epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 4,
        "learning_rate": 5e-6,
        "max_length": 512,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset)
    }
    with open(os.path.join(out_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    start_time = time.time()
    try:
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        
        print("Starting 3-epoch candidate training...")
        train_result = trainer.train()
        train_time = time.time() - start_time
        peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"Training finished in {train_time:.2f} seconds. Peak VRAM: {peak_vram:.2f} GB")
        
        print("Running final validation evaluation...")
        eval_metrics = trainer.evaluate()
        print("Validation metrics:", eval_metrics)
        
        print(f"Saving 3-epoch LoRA adapter to {out_dir}...")
        trainer.model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        
        metrics_summary = {
            "train_runtime_seconds": train_time,
            "train_loss": train_result.training_loss,
            "train_steps": train_result.global_step,
            "peak_vram_gb": round(peak_vram, 3),
            "log_history": trainer.state.log_history,
            "eval_metrics": eval_metrics
        }
        with open(os.path.join(out_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
            
        print("3-epoch DPO training successfully completed!")
        
    except Exception as e:
        print(f"Training or OOM Error: {e}")
        import traceback
        traceback.print_exc()
        print("Peak VRAM at failure:", torch.cuda.max_memory_allocated(0) / 1024**3, "GB")
        
    finally:
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
