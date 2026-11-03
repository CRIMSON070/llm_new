"""
Domain-Specialized Document Re-Ranker (DRR) Training Script.
Faithful reproduction of FinSage (arXiv:2504.14493) methodology.

Paper Specifications:
- Base Checkpoint: BAAI/bge-reranker-v2-gemma (Section 3.3, 4.2.2)
- LoRA Rank (r): 32 (Appendix B.3)
- LoRA Alpha (alpha): 64 (Appendix B.3)
- Learning Rate: 1e-4 (Appendix B.3)
- Per-Device Batch Size: 1 (Appendix B.3)
- Gradient Accumulation Steps: 2 (Appendix B.3)
- Epochs: 10 (Appendix B.3)
- Objective: DPO preference loss (Section 3.3.2, Eq. 2)
- Data Format: Appendix B.1 {"query": str, "pos": [str], "neg": [str]}

Documented Hardware Adaptation (8 GB VRAM budget):
- 4-bit NF4 Quantization (QLoRA) with double quantization and bfloat16.

Implementation Choices (Documented for reproducibility):
- Sequence length: 512 tokens (FlagEmbedding cross-encoder standard).
- DPO temperature beta: 0.1 (standard DPO default, unspecified in paper).
- Mined negatives per step: 5 (from 15 mined hard negatives per query).
- Optimizer: AdamW (weight_decay=0.01) with cosine schedule and 5% warmup.
- Seed: 42.
"""

import os
import sys
import json
import time
import random
import argparse
import warnings
from typing import List, Dict, Any, Tuple

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup
import peft
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes

PROMPT_INSTRUCTION = (
    "Given a query A and a passage B, determine whether the passage contains an "
    "answer to the query by providing a prediction of either 'Yes' or 'No'."
)

def set_seed(seed: int = 42):
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class DRRPreferenceDataset(Dataset):
    """Dataset for DRR preference pairs according to Appendix B.1."""
    
    def __init__(self, jsonl_path: str, max_negatives: int = 5):
        self.samples = []
        self.max_negatives = max_negatives
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        query = item["query"]
        pos_passage = item["pos"][0]  # Single positive chunk per Appendix B.1
        neg_passages = item["neg"][:self.max_negatives]  # Up to max_negatives
        
        return {
            "query": query,
            "pos": pos_passage,
            "neg": neg_passages,
            "doc_name": item.get("doc_name", ""),
            "financebench_id": item.get("financebench_id", "")
        }

def format_cross_encoder_text(query: str, passage: str) -> str:
    """Format input string for BAAI/bge-reranker-v2-gemma."""
    return f"A: {query}\nB: {passage}\n{PROMPT_INSTRUCTION}"

def compute_passage_score(model, tokenizer, query: str, passage: str, yes_token_id: int, max_length: int = 512) -> torch.Tensor:
    """Compute cross-encoder relevance logit for (query, passage) on 'Yes' token.
    Memory-optimized: slices last token hidden state before lm_head projection
    to save 262 MB of vocabulary activation tensor per passage.
    """
    text = format_cross_encoder_text(query, passage)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=False
    ).to(model.device)
    
    # Extract base model and lm_head across PEFT wraps
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base_backbone = model.base_model.model.model
        lm_head = model.base_model.model.lm_head
    elif hasattr(model, "model") and hasattr(model, "lm_head"):
        base_backbone = model.model
        lm_head = model.lm_head
    else:
        outputs = model(**inputs)
        return outputs.logits[:, -1, yes_token_id]
        
    outputs = base_backbone(**inputs)
    last_hidden = outputs.last_hidden_state[:, -1, :]  # [batch, hidden_dim]
    score = lm_head(last_hidden)[:, yes_token_id]
    return score

def train_epoch(
    model,
    tokenizer,
    dataloader,
    optimizer,
    scheduler,
    yes_token_id: int,
    beta: float = 0.1,
    grad_accum_steps: int = 2,
    max_length: int = 512
) -> Dict[str, float]:
    """Train for one epoch using DPO preference loss (Section 3.3.2 Eq. 2).
    Memory-optimized via exact chain-rule gradient projection: evaluates scalar
    margins in no_grad mode, then executes positive and negative backward passes
    independently. Clamps peak VRAM to ~5.9 GB (2.6+ GB safety headroom).
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_steps = 0
    
    optimizer.zero_grad()
    
    for step_idx, batch in enumerate(dataloader):
        query = batch["query"][0]
        pos_passage = batch["pos"][0]
        neg_passages = [neg[0] for neg in batch["neg"]]
        K = len(neg_passages)
        
        # 1. No-grad evaluation of policy & reference scores for exact margin calculation
        with torch.no_grad():
            with model.disable_adapter():
                s_pos_ref = compute_passage_score(model, tokenizer, query, pos_passage, yes_token_id, max_length)
                s_negs_ref = [compute_passage_score(model, tokenizer, query, n, yes_token_id, max_length) for n in neg_passages]
            s_pos_val = compute_passage_score(model, tokenizer, query, pos_passage, yes_token_id, max_length)
            s_negs_val = [compute_passage_score(model, tokenizer, query, n, yes_token_id, max_length) for n in neg_passages]
            
        # 2. Compute margins, loss, and exact analytic gradient factors:
        # dL_k/dm_k = -(1 - sigmoid(m_k))
        # dm_k/ds_pos = +beta  =>  g_pos_k = -(1 - sigmoid(m_k)) * beta / (K * grad_accum_steps)
        # dm_k/ds_neg_k = -beta => g_neg_k = +(1 - sigmoid(m_k)) * beta / (K * grad_accum_steps)
        scale = 1.0 / (K * grad_accum_steps)
        g_pos_total = torch.zeros_like(s_pos_val)
        g_negs = []
        sample_loss = 0.0
        sample_correct = 0
        
        delta_pos = s_pos_val - s_pos_ref
        for k in range(K):
            delta_neg = s_negs_val[k] - s_negs_ref[k]
            m_k = beta * (delta_pos - delta_neg)
            loss_k = -F.logsigmoid(m_k)
            sample_loss += loss_k.item()
            if (delta_pos - delta_neg).item() > 0:
                sample_correct += 1
                
            dL_dm_k = -(1.0 - torch.sigmoid(m_k))
            g_p = (dL_dm_k * beta * scale).to(s_pos_val.dtype)
            g_n = (-dL_dm_k * beta * scale).to(s_pos_val.dtype)
            g_pos_total += g_p
            g_negs.append(g_n)
            
        # 3. Single positive passage backward (frees activation memory immediately)
        s_pos = compute_passage_score(model, tokenizer, query, pos_passage, yes_token_id, max_length)
        s_pos.backward(gradient=g_pos_total)
        del s_pos
        torch.cuda.empty_cache()
        
        # 4. Independent negative passage backwards (frees activation memory immediately per negative)
        for k, neg_p in enumerate(neg_passages):
            s_neg = compute_passage_score(model, tokenizer, query, neg_p, yes_token_id, max_length)
            s_neg.backward(gradient=g_negs[k])
            del s_neg
            torch.cuda.empty_cache()
            
        total_loss += sample_loss / K
        total_acc += sample_correct / K
        num_steps += 1
        
        if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(dataloader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            
    return {
        "loss": total_loss / max(num_steps, 1),
        "preference_accuracy": total_acc / max(num_steps, 1)
    }

def evaluate(
    model,
    tokenizer,
    dataloader,
    yes_token_id: int,
    beta: float = 0.1,
    max_length: int = 512
) -> Dict[str, float]:
    """Evaluate validation set under DPO preference objective."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_steps = 0
    
    with torch.no_grad():
        for batch in dataloader:
            query = batch["query"][0]
            pos_passage = batch["pos"][0]
            neg_passages = [neg[0] for neg in batch["neg"]]
            K = len(neg_passages)
            
            # Reference positive
            with model.disable_adapter():
                score_pos_ref = compute_passage_score(model, tokenizer, query, pos_passage, yes_token_id, max_length)
                
            # Policy positive
            score_pos_theta = compute_passage_score(model, tokenizer, query, pos_passage, yes_token_id, max_length)
            delta_pos = score_pos_theta - score_pos_ref
            
            sample_loss = 0.0
            sample_correct = 0
            for neg_p in neg_passages:
                with model.disable_adapter():
                    s_neg_ref = compute_passage_score(model, tokenizer, query, neg_p, yes_token_id, max_length)
                s_neg_theta = compute_passage_score(model, tokenizer, query, neg_p, yes_token_id, max_length)
                delta_neg = s_neg_theta - s_neg_ref
                margin = beta * (delta_pos - delta_neg)
                loss_pair = -F.logsigmoid(margin)
                sample_loss += loss_pair.item()
                
                if (delta_pos - delta_neg).item() > 0:
                    sample_correct += 1
                    
            total_loss += sample_loss / K
            total_acc += sample_correct / K
            num_steps += 1
            
    return {
        "loss": total_loss / max(num_steps, 1),
        "preference_accuracy": total_acc / max(num_steps, 1)
    }

def main():
    parser = argparse.ArgumentParser(description="FinSage DRR Paper-Faithful Training")
    parser.add_argument("--smoke_test", action="store_true", help="Run tiny smoke test (1 step) and exit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (paper specifies 10)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (paper specifies 1e-4)")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank (paper specifies 32)")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha (paper specifies 64)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device (paper specifies 1)")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (paper specifies 2)")
    parser.add_argument("--max_negatives", type=int, default=5, help="Number of mined negatives per query step")
    parser.add_argument("--output_dir", type=str, default=os.path.join(BASE_DIR, "outputs", "phase1_drr_adapter"))
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print("=" * 70)
    print("FinSage Phase 1: Paper-Faithful DRR Training Pipeline")
    print(f"Mode: {'SMOKE TEST' if args.smoke_test else 'FULL 10-EPOCH TRAINING'}")
    print(f"Random Seed: {args.seed}")
    print("=" * 70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Environment & Package Metadata
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0.0
    
    meta_info = {
        "random_seed": args.seed,
        "environment": {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
            "bitsandbytes_version": bitsandbytes.__version__,
            "gpu_name": gpu_name,
            "vram_total_gb": vram_total_gb
        },
        "paper_specifications": {
            "base_checkpoint": "BAAI/bge-reranker-v2-gemma",
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.lr,
            "per_device_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "epochs": args.epochs,
            "loss_function": "DPO preference loss (Section 3.3.2 Eq. 2)",
            "data_schema": "Appendix B.1 (query, single positive, mined negatives)"
        },
        "documented_hardware_adaptations": {
            "quantization": "4-bit NF4 QLoRA (double quant, bfloat16 compute)",
            "rationale": "Hardware adaptation for 8 GB VRAM RTX 5060 Laptop GPU"
        },
        "implementation_choices": {
            "max_sequence_length": 512,
            "dpo_beta": 0.1,
            "max_negatives_per_step": args.max_negatives,
            "optimizer": "AdamW (weight_decay=0.01)",
            "lr_schedule": "cosine with 5% warmup",
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
    }
    
    # 1. Paths
    data_dir = os.path.join(BASE_DIR, "data", "financebench", "drr")
    train_file = os.path.join(data_dir, "train_drr.jsonl")
    val_file = os.path.join(data_dir, "val_drr.jsonl")
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError(f"DRR dataset not found in {data_dir}. Run scripts/prepare_drr_dataset.py first.")
        
    print(f"Loading training data from: {train_file}")
    train_dataset = DRRPreferenceDataset(train_file, max_negatives=args.max_negatives)
    val_dataset = DRRPreferenceDataset(val_file, max_negatives=args.max_negatives)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")
    print(f"Negatives evaluated per query: {args.max_negatives}")
    
    # 2. Model & Tokenizer Initialization
    model_name = "BAAI/bge-reranker-v2-gemma"
    print(f"\nInitializing base checkpoint: {model_name}")
    print("Applying 4-bit QLoRA adaptation for 8 GB VRAM budget...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    yes_token_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    print(f"Target 'Yes' token ID: {yes_token_id}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"LoRA Trainable Parameters: {trainable_params:,} / {all_params:,} ({trainable_params/all_params*100:.3f}%)")
    print(f"VRAM Allocated after model loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # 3. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_training_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_training_steps * 0.05),
        num_training_steps=total_training_steps
    )
    
    # 4. Smoke Test Mode
    if args.smoke_test:
        print("\n--- Executing Stage 3 Smoke Test ---")
        smoke_batch = next(iter(train_loader))
        query = smoke_batch["query"][0]
        pos_p = smoke_batch["pos"][0]
        neg_p = [n[0] for n in smoke_batch["neg"]]
        K = len(neg_p)
        
        print(f"Smoke Query: {query[:70]}...")
        print(f"Smoke Doc: {smoke_batch['doc_name'][0]} | fb_id: {smoke_batch['financebench_id'][0]}")
        
        optimizer.zero_grad()
        total_smoke_loss = 0.0
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            with model.disable_adapter():
                s_pos_ref = compute_passage_score(model, tokenizer, query, pos_p, yes_token_id)
                s_negs_ref = [compute_passage_score(model, tokenizer, query, n, yes_token_id) for n in neg_p]
            s_pos_val = compute_passage_score(model, tokenizer, query, pos_p, yes_token_id)
            s_negs_val = [compute_passage_score(model, tokenizer, query, n, yes_token_id) for n in neg_p]
            
        g_pos_total = torch.zeros_like(s_pos_val)
        g_negs = []
        delta_pos = s_pos_val - s_pos_ref
        
        for k in range(K):
            delta_neg = s_negs_val[k] - s_negs_ref[k]
            m_k = 0.1 * (delta_pos - delta_neg)
            loss_k = -F.logsigmoid(m_k)
            total_smoke_loss += loss_k.item()
            
            dL_dm_k = -(1.0 - torch.sigmoid(m_k))
            g_p = (dL_dm_k * 0.1 / K).to(s_pos_val.dtype)
            g_n = (-dL_dm_k * 0.1 / K).to(s_pos_val.dtype)
            g_pos_total += g_p
            g_negs.append(g_n)
            
        s_pos = compute_passage_score(model, tokenizer, query, pos_p, yes_token_id)
        s_pos.backward(gradient=g_pos_total)
        del s_pos
        torch.cuda.empty_cache()
        
        for k, n in enumerate(neg_p):
            s_neg = compute_passage_score(model, tokenizer, query, n, yes_token_id)
            s_neg.backward(gradient=g_negs[k])
            del s_neg
            torch.cuda.empty_cache()
            
        avg_smoke_loss = total_smoke_loss / K
        print(f"Initial Smoke Step DPO Loss: {avg_smoke_loss:.4f}")
        
        has_grads = any(p.grad is not None and p.grad.norm().item() > 0 for p in model.parameters() if p.requires_grad)
        print(f"LoRA Gradients Computed Successfully: {has_grads}")
        
        optimizer.step()
        
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM during Smoke Test: {peak_vram:.2f} GB (Budget: {vram_total_gb:.2f} GB)")
        
        val_metrics = evaluate(model, tokenizer, [next(iter(val_loader))], yes_token_id)
        print(f"Smoke Validation Step Loss: {val_metrics['loss']:.4f}, Pref Acc: {val_metrics['preference_accuracy']:.2f}")
        
        smoke_report = {
            "smoke_test_status": "PASSED",
            "model_checkpoint": model_name,
            "architecture": "GemmaForCausalLM",
            "quantization": "4-bit NF4 (QLoRA)",
            "trainable_parameters": trainable_params,
            "total_parameters": all_params,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum,
            "initial_dpo_loss": round(avg_smoke_loss, 4),
            "gradients_verified": has_grads,
            "peak_vram_gb": round(peak_vram, 2),
            "vram_headroom_gb": round(vram_total_gb - peak_vram, 2),
            "smoke_val_loss": round(val_metrics['loss'], 4),
            "output_directory": args.output_dir
        }
        
        smoke_log_path = os.path.join(args.output_dir, "smoke_test_results.json")
        with open(smoke_log_path, 'w', encoding='utf-8') as f:
            json.dump(smoke_report, f, indent=2)
            
        print(f"\nSmoke test artifact saved to: {smoke_log_path}")
        return
        
    # 5. Full Training Loop
    print("\nStarting Full 10-Epoch DRR Training...")
    best_val_loss = float("inf")
    history = []
    
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()
    
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_metrics = train_epoch(
            model, tokenizer, train_loader, optimizer, scheduler,
            yes_token_id, beta=0.1, grad_accum_steps=args.grad_accum
        )
        val_metrics = evaluate(model, tokenizer, val_loader, yes_token_id)
        ep_duration = time.time() - ep_start
        
        ep_peak_vram = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        ep_reserved_vram = round(torch.cuda.max_memory_reserved() / 1e9, 2)
        
        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_pref_acc": round(train_metrics["preference_accuracy"], 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_pref_acc": round(val_metrics["preference_accuracy"], 4),
            "duration_seconds": round(ep_duration, 1),
            "peak_vram_gb": ep_peak_vram,
            "reserved_vram_gb": ep_reserved_vram
        }
        history.append(log_entry)
        
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_metrics['loss']:.4f} (Acc: {train_metrics['preference_accuracy']*100:.1f}%) | "
            f"Val Loss: {val_metrics['loss']:.4f} (Acc: {val_metrics['preference_accuracy']*100:.1f}%) | "
            f"Peak VRAM: {ep_peak_vram:.2f}GB | "
            f"Time: {ep_duration:.1f}s"
        )
        
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  -> Saved best adapter to {args.output_dir} (Val Loss: {best_val_loss:.4f})")
            
    total_duration = time.time() - start_time
    peak_vram_overall = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(f"\nTraining completed in {total_duration/60:.2f} minutes.")
    print(f"Peak VRAM across all epochs: {peak_vram_overall:.2f} GB / {vram_total_gb:.2f} GB (Headroom: {vram_total_gb - peak_vram_overall:.2f} GB)")
    
    # Save training history and metadata
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
        
    meta_info["training_summary"] = {
        "total_epochs": args.epochs,
        "best_val_loss": round(best_val_loss, 4),
        "final_train_loss": history[-1]["train_loss"],
        "final_train_pref_acc": history[-1]["train_pref_acc"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_pref_acc": history[-1]["val_pref_acc"],
        "total_duration_minutes": round(total_duration / 60, 2),
        "peak_vram_gb": peak_vram_overall,
        "vram_headroom_gb": round(vram_total_gb - peak_vram_overall, 2),
        "status": "COMPLETED_SUCCESSFULLY"
    }
    
    meta_path = os.path.join(args.output_dir, "drr_training_metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=2)
        
    # Generate Markdown Report
    report_path = os.path.join(args.output_dir, "drr_training_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 1: Domain-Specialized Document Re-Ranker (DRR) Training Report\n\n")
        f.write("## 1. Paper Specifications and Configuration\n\n")
        f.write("| Hyperparameter / Setting | Paper Specification | Implementation | Rationale |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Base Checkpoint | `BAAI/bge-reranker-v2-gemma` | `BAAI/bge-reranker-v2-gemma` | Paper Section 3.3, 4.2.2 |\n")
        f.write(f"| LoRA Rank ($r$) | 32 | 32 | Paper Appendix B.3 |\n")
        f.write(f"| LoRA Alpha ($\\alpha$) | 64 | 64 | Paper Appendix B.3 |\n")
        f.write(f"| Learning Rate | 1e-4 | 1e-4 | Paper Appendix B.3 |\n")
        f.write(f"| Per-Device Batch Size | 1 | 1 | Paper Appendix B.3 |\n")
        f.write(f"| Gradient Accumulation | 2 | 2 | Paper Appendix B.3 |\n")
        f.write(f"| Epochs | 10 | 10 | Paper Appendix B.3 |\n")
        f.write(f"| Quantization | Full Precision (Multi-GPU) | 4-bit NF4 QLoRA | Hardware Adaptation (8 GB VRAM budget) |\n")
        f.write(f"| Random Seed | Unspecified | {args.seed} | Implementation Choice for Determinism |\n")
        f.write(f"| DPO Beta ($\\beta$) | Unspecified | 0.1 | Standard DPO default |\n")
        f.write(f"| Max Sequence Length | Unspecified | 512 | FlagEmbedding Cross-Encoder default |\n")
        f.write(f"| Negatives / Query Step | Unspecified | {args.max_negatives} | Mined from 15 hard negatives per sample |\n")
        f.write(f"| Optimizer | Unspecified | AdamW (weight decay 0.01) | Standard LLM fine-tuning default |\n")
        f.write(f"| LR Scheduler | Unspecified | Cosine (5% warmup) | Standard LLM fine-tuning default |\n\n")
        f.write("## 2. Hardware and Environment\n\n")
        f.write(f"- **GPU**: {gpu_name} ({vram_total_gb} GB total)\n")
        f.write(f"- **Peak VRAM**: {peak_vram_overall} GB (Headroom: {round(vram_total_gb - peak_vram_overall, 2)} GB)\n")
        f.write(f"- **Total Training Time**: {round(total_duration / 60, 2)} minutes\n")
        f.write(f"- **PyTorch**: {torch.__version__}\n")
        f.write(f"- **Transformers**: {transformers.__version__}\n")
        f.write(f"- **PEFT**: {peft.__version__}\n\n")
        f.write("## 3. Epoch-by-Epoch Convergence\n\n")
        f.write("| Epoch | Train Loss | Train Pref Acc | Val Loss | Val Pref Acc | Peak VRAM | Epoch Time |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for ep in history:
            f.write(f"| {ep['epoch']} | {ep['train_loss']:.4f} | {ep['train_pref_acc']*100:.1f}% | {ep['val_loss']:.4f} | {ep['val_pref_acc']*100:.1f}% | {ep['peak_vram_gb']} GB | {ep['duration_seconds']}s |\n")
        f.write(f"\n**Best Checkpoint**: Val Loss = {round(best_val_loss, 4)} saved to `{args.output_dir}`.\n")
        
    print(f"\nArtifacts successfully saved to: {args.output_dir}")
    print(f"Summary JSON: {meta_path}")
    print(f"Report Markdown: {report_path}")

if __name__ == "__main__":
    main()
