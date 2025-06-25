#!/usr/bin/env python
"""Generate text from triples using trained LoRA adapters.
Usage:
    python finetune_generate.py --lang en --model webnlg_llama2_13b_en \
        --triples "London|located_in|England"
"""
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

CFG_BASE = {
    "en": "meta-llama/Llama-2-13b-chat-hf",
    "ga": "ReliableAI/UCCIX-Llama2-13B-Instruct-191224",
}

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--lang", choices=["en", "ga"], default="en")
    pa.add_argument("--model", required=True, help="Folder with trained adapters & tokenizer")
    pa.add_argument("--triples", required=True, help="Pipe-separated triple string")
    pa.add_argument("--max_new_tokens", type=int, default=128)
    pa.add_argument("--temperature", type=float, default=0.7)
    pa.add_argument("--top_p", type=float, default=0.9)
    args = pa.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    backbone = AutoModelForCausalLM.from_pretrained(CFG_BASE[args.lang], torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(backbone, args.model)

    user = f"Convert the following triples into a fluent {'Irish' if args.lang=='ga' else 'English'} description:\n{args.triples}"
    prompt = f"<s>[INST] {user} [/INST]"

    outs = model.generate(**tok(prompt, return_tensors="pt").to(model.device),
                          max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                          top_p=args.top_p, do_sample=True, eos_token_id=tok.eos_token_id)
    print(tok.decode(outs[0], skip_special_tokens=True).split("[/INST]")[-1].strip())

if __name__ == "__main__":
    main()