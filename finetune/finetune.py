#!/usr/bin/env python

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

###########################
# 1.  Data extraction     #
###########################

def extract_cmd(args: argparse.Namespace) -> None:
    """Parse ga_train.xml → JSONL with list[dict(triples, text)]"""
    entries = []
    root = ET.parse(args.input).getroot()

    for entry in root.iter("entry"):
        # Prefer <modifiedtripleset>; fall back to <originaltripleset>
        mset = entry.find("modifiedtripleset") or entry.find("originaltripleset")
        if mset is None:
            continue  # malformed entry
        triples = [t.text.strip() for t in mset]

        # Multiple lexicalisations per entry
        for lex in entry.findall("lex"):
            lang = lex.attrib.get("lang", "ga").lower()
            if not lang.startswith("ga"):
                continue  # skip non‑Irish
            text = " ".join((lex.text or "").split())  # normalise whitespace
            entries.append({"triples": triples, "text": text})

    # Write JSONL so we can stream‑load later
    with open(args.output, "w", encoding="utf‑8") as fh:
        for item in entries:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[extract] Saved {len(entries)} examples → {args.output}")

###########################
# 2.  LoRA fine‑tuning    #
###########################

def finetune_cmd(args: argparse.Namespace) -> None:
    """LoRA‑fine‑tune ReliableAI/UCCIX‑Llama2‑13B‑Instruct‑191224 on Irish data"""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    # ---------- Build dataset ----------
    def prompt_template(rec):
        return f"Triple: {' | '.join(rec['triples'])}\n### Response (ga): {rec['text']}"

    records = [json.loads(l) for l in Path(args.train).read_text(encoding="utf‑8").splitlines()]
    ds = Dataset.from_list([{"prompt": prompt_template(r)} for r in records])

    # ---------- Tokeniser / model ----------
    base_id = "ReliableAI/UCCIX-Llama2-13B-Instruct-191224"
    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # Llama‑style models have no pad token

    ds_tok = ds.map(
        lambda ex: tokenizer(ex["prompt"], truncation=True, padding="max_length", max_length=1024),
        batched=True,
        remove_columns=["prompt"],
    )

    model = AutoModelForCausalLM.from_pretrained(base_id, load_in_8bit=True, device_map="auto")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # works well for Llama‑family
    )
    model = get_peft_model(model, lora_cfg)

    # ---------- Trainer ----------
    targs = TrainingArguments(
        output_dir=args.outdir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        save_steps=500,
        save_total_limit=2,
        logging_steps=25,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tok,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    # ---------- Save ----------
    model.save_pretrained(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    print(f"[finetune] LoRA adapters + tokenizer saved → {args.outdir}")

###########################
# 3.  Irish generation    #
###########################

# def generate_cmd(args: argparse.Namespace) -> None:
#     """Generate Gaelic text from a pipe‑separated triple string using the tuned adapters"""
#     import torch
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     from peft import PeftModel

#     base_id = "ReliableAI/UCCIX-Llama2-13B-Instruct-191224"
#     tokenizer = AutoTokenizer.from_pretrained(args.model)
#     base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16, device_map="auto")
#     model = PeftModel.from_pretrained(base, args.model)

#     triples = [t.strip() for t in args.triples.split("|")]
#     prompt = f"Triple: {' | '.join(triples)}\n### Response (ga):"

#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=args.max_new_tokens,
#         temperature=args.temperature,
#         top_p=args.top_p,
#         do_sample=True,
#         eos_token_id=tokenizer.eos_token_id,
#     )

#     text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     generated = text.split("### Response (ga):")[-1].strip()
#     print(generated)

###########################
# 4.  CLI entry‑point     #
###########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PEFT LoRA fine‑tune UCCIX on WebNLG‑Irish data")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # p1 = sub.add_parser("extract", help="Convert ga_train.xml → JSONL")
    # p1.add_argument("--input", default="ga_train.xml", help="Path to ga_train.xml")
    # p1.add_argument("--output", default="ga_train_parsed.jsonl", help="Output JSONL file")
    # p1.set_defaults(func=extract_cmd)

    p2 = sub.add_parser("finetune", help="LoRA fine‑tune step")
    p2.add_argument("--train", default="ga_train_parsed.jsonl", help="Training JSONL file")
    p2.add_argument("--outdir", default="./uccix_ga_lora", help="Where to save adapters & tokenizer")
    p2.set_defaults(func=finetune_cmd)

    # p3 = sub.add_parser("generate", help="Generate Gaelic text from triples")
    # p3.add_argument("--model", default="./uccix_ga_lora", help="LoRA adapter directory")
    # p3.add_argument("--triples", required=True, help="Pipe‑separated triple string, e.g. 'Athlone|located_in|Ireland'")
    # p3.add_argument("--max_new_tokens", type=int, default=128)
    # p3.add_argument("--temperature", type=float, default=0.7)
    # p3.add_argument("--top_p", type=float, default=0.9)
    # p3.set_defaults(func=generate_cmd)

    args = parser.parse_args()
    args.func(args)

