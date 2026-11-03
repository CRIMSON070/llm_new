"""
FinSage Generative HyDE Module (Path 4 of Multi-Path Retrieval)
==============================================================
Paper Specifications: FinSage (arXiv:2504.14493, CIKM 2025) Section 3.2 & 4.2.1.

Generates 3 hypothetical document passages per query using Qwen2.5-7B:
- Uses the paper's exact system prompt from src/prompts/hyde.py
- Generates 2 text passages + 1 structured table ([Table Level] + [Row Level])
- Embeds hypothetical passages with BAAI/bge-m3
- Retrieves dense candidate chunks from FAISS
"""

import os
import sys
import re
import requests
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.prompts.hyde import get_hypo_sys_prompt


class GenerativeHyDE:
    """
    Paper-faithful Generative HyDE implementation for FinSage.
    Generates 3 hypothetical document answers using Qwen2.5-7B.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        api_base: str = "http://localhost:11434/v1",
        hf_model=None,
        hf_tokenizer=None,
    ):
        self.model_name = model_name
        self.api_base = api_base
        self.hf_model = hf_model
        self.hf_tokenizer = hf_tokenizer
        self.use_api = True

        # Test API connection
        try:
            test_resp = requests.get(f"{self.api_base.replace('/v1', '')}/api/tags", timeout=3)
            if test_resp.status_code == 200:
                print(f"[GenerativeHyDE] Connected to local Ollama API for {model_name}.")
                self.use_api = True
            else:
                self.use_api = False
        except Exception:
            self.use_api = False

        if not self.use_api and self.hf_model is None:
            print(f"[GenerativeHyDE] Fallback to Hugging Face Transformers...")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct",
                quantization_config=quant_config,
                device_map="auto"
            )
            self.hf_model.eval()

    def generate_hypothetical_passages(
        self,
        query: str,
        num_hypo: int = 3,
        max_new_tokens: int = 320,
        temperature: float = 0.3,
    ) -> List[str]:
        """
        Generates num_hypo (default 3) hypothetical document passages based on
        the paper's exact system prompt in src/prompts/hyde.py.
        """
        sys_prompt = get_hypo_sys_prompt(num_hypo=num_hypo)

        if self.use_api:
            try:
                resp = requests.post(
                    f"{self.api_base}/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": query}
                        ],
                        "temperature": temperature,
                        "max_tokens": max_new_tokens
                    },
                    timeout=45
                )
                if resp.status_code == 200:
                    gen_text = resp.json()["choices"][0]["message"]["content"].strip()
                else:
                    gen_text = query
            except Exception as e:
                print(f"[GenerativeHyDE] API generation error: {e}")
                gen_text = query
        else:
            import torch
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query}
            ]
            prompt_text = self.hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.hf_tokenizer(prompt_text, return_tensors="pt").to(self.hf_model.device)
            with torch.no_grad():
                outputs = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0.0),
                    top_p=0.8,
                    pad_token_id=self.hf_tokenizer.eos_token_id
                )
            gen_text = self.hf_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

        # Parse ANSWER blocks
        passages = []
        if "ANSWER:" in gen_text:
            parts = gen_text.split("ANSWER:")
            for p in parts[1:]:
                cleaned = p.strip()
                if cleaned:
                    passages.append(cleaned)

        if not passages:
            passages = [gen_text]

        return passages[:num_hypo]
