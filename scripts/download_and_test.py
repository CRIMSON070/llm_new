import os
import torch
import gc
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Starting download for {model_id}...")
    
    try:
        # Download the model files with retry and resume enabled
        cached_path = snapshot_download(
            repo_id=model_id,
            resume_download=True,
            max_workers=1,
            local_files_only=False
        )
        print(f"Download complete. Model files are located at: {cached_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        return

    print("\nTesting 4-bit NF4 Loading...")
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot test 4-bit load.")
        return
        
    print(f"Initial VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("Tokenizer loaded.")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto"
        )
        print("Model loaded successfully in 4-bit!")
        
        vram_allocated = torch.cuda.memory_allocated(0) / 1024**3
        vram_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"VRAM Allocated: {vram_allocated:.2f} GB")
        print(f"VRAM Reserved: {vram_reserved:.2f} GB")
        
    except Exception as e:
        print(f"Error loading model: {e}")
    finally:
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("VRAM cleared.")

if __name__ == "__main__":
    main()
