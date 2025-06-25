#!/usr/bin/env python
"""LoRA‑fine‑tune WebNLG EN/GA dataset on a 13‑B Llama‑family model.
   * English → meta‑llama/Llama‑2‑13b‑chat‑hf
   * Irish   → ReliableAI/UCCIX‑Llama2‑13B‑Instruct‑191224

Example:
    python finetune_model.py --lang en --train en_train.jsonl --outdir webnlg_llama2_13b_en
"""
import argparse, json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
                          Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, TaskType

CFG = {
    "en": {
        "base": "meta-llama/Llama-2-13b-chat-hf",
        "sys": "You are a helpful data-to-text generation assistant.",
        "out": "webnlg_llama2_13b_en",
    },
    "ga": {
        "base": "ReliableAI/UCCIX-Llama2-13B-Instruct-191224",
        "sys": "Is cúntóir giniúna sonraí-go-téacs cabhrach thú.",
        "out": "webnlg_llama2_13b_ga",
    },
}

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--lang", choices=["en", "ga"], default="en")
    pa.add_argument("--train", required=True)
    pa.add_argument("--outdir")
    args = pa.parse_args()

    cfg = CFG[args.lang]
    outdir = args.outdir or cfg["out"]

    # Build dataset → chat prompt
    def build_prompt(rec):
        triples = " | ".join(rec["triples"])
        user = f"Convert the following triples into a fluent {'Irish' if args.lang=='ga' else 'English'} description:\n{triples}"
        return f"<s>[INST] <<SYS>>\n{cfg['sys']}\n<</SYS>>\n\n{user} [/INST] {rec['text']} </s>"

    records = [json.loads(l) for l in Path(args.train).read_text("utf-8").splitlines()]
    ds = Dataset.from_list([{"text": build_prompt(r)} for r in records])

    tok = AutoTokenizer.from_pretrained(cfg["base"], use_fast=False)
    tok.pad_token = tok.eos_token
    ds_tok = ds.map(lambda ex: tok(ex["text"], truncation=True, padding="max_length", max_length=1024),
                    batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(cfg["base"], load_in_8bit=True, device_map="auto")
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora_cfg)

    targs = TrainingArguments(output_dir=outdir, num_train_epochs=3,
                              per_device_train_batch_size=4, gradient_accumulation_steps=4,
                              learning_rate=2e-4, fp16=torch.cuda.is_available(),
                              save_steps=500, save_total_limit=2, logging_steps=25, report_to="none")

    Trainer(model=model, args=targs, train_dataset=ds_tok,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()

    model.save_pretrained(outdir); tok.save_pretrained(outdir)
    print(f"[finetune] LoRA adapters saved → {outdir}")

if __name__ == "__main__":
    main()