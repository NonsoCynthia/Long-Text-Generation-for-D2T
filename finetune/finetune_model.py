#!/usr/bin/env python
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import torch
from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    PromptTuningConfig, PromptTuningInit,
    PrefixTuningConfig
)

# ---------- Utility Functions ----------
def set_seed(seed: int):
    import random, numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_triples(row: dict, triples_div: str = "; ") -> dict:
    triples = row.get("input")
    if isinstance(triples, list):
        triples_str = triples_div.join(triples)
    elif isinstance(triples, str):
        triples_str = triples.replace(" | ", triples_div)
    else:
        triples_str = ""
    # Always output "target" as "text" for consistency downstream
    text = row.get("target") or row.get("text") or (row["references"][0] if "references" in row else "")
    return {"triples": triples_str, "text": text}


def formatting_func(samples: dict, lang: str, template: str) -> List[str]:
    lang_str = "Irish" if lang == "ga" else "English"
    return [
        template.format(lang=lang_str, triples=triples, output=text)
        for triples, text in zip(samples["triples"], samples["text"])
    ]

# --------------- Configuration -----------------
@dataclass
class ScriptArguments:
    seed: Optional[int] = field(default=6787)
    lang: Optional[str] = field(default="en")
    peft_type: Optional[str] = field(default="lora")
    train_file: Optional[str] = field(default=None)
    dev_file: Optional[str] = field(default=None)
    outdir: Optional[str] = field(default="webnlg_llama2_13b_en_lora")
    template: Optional[str] = field(
        default="[INST]Convert the following triples into a fluent {lang} description:\n{triples}[/INST][ANS]{output}[/ANS]"
    )
    per_device_train_batch_size: Optional[int] = 4
    gradient_accumulation_steps: Optional[int] = 4
    learning_rate: Optional[float] = 2e-4
    max_seq_length: Optional[int] = 512
    num_train_epochs: Optional[int] = 3
    max_steps: Optional[int] = 10000
    num_virtual_tokens: Optional[int] = 8
    prefix_projection: Optional[bool] = False
    save_steps: Optional[int] = 500
    save_total_limit: Optional[int] = 10
    logging_steps: Optional[int] = 25
    lora_alpha: Optional[int] = 16
    lora_dropout: Optional[float] = 0.1
    lora_r: Optional[int] = 64
    max_grad_norm: Optional[float] = 0.3
    weight_decay: Optional[float] = 0.001
    use_4bit: Optional[bool] = True
    warmup_ratio: Optional[float] = 0.3
    base_model: Optional[str] = field(default="meta-llama/Llama-2-13b-chat-hf")

def parse_args():
    pa = argparse.ArgumentParser()
    for f in ScriptArguments.__dataclass_fields__:
        t = ScriptArguments.__dataclass_fields__[f].type
        default = ScriptArguments.__dataclass_fields__[f].default
        if t is bool or t is Optional[bool]:
            pa.add_argument(f"--{f}", action="store_true" if not default else "store_false")
        else:
            pa.add_argument(f"--{f}", type=type(default) if default is not None else str, default=default)
    return pa.parse_args()

def main():
    args_ns = parse_args()
    script_args = ScriptArguments(**vars(args_ns))
    set_seed(script_args.seed)

    if script_args.lang == "en":
        print("[data] Loading GEM/web_nlg English split")
        ds = load_dataset("GEM/web_nlg", "en")
        train = ds["train"].map(preprocess_triples)
        dev = ds["validation"].map(preprocess_triples)
    else:
        assert script_args.train_file and script_args.dev_file, "Provide --train_file and --dev_file for non-English."
        train = load_dataset("json", data_files=script_args.train_file, split="train").map(preprocess_triples)
        dev = load_dataset("json", data_files=script_args.dev_file, split="train").map(preprocess_triples)

        train_prompts = formatting_func(
            {"triples": [r["triples"] for r in train], "text": [r["target"] for r in train]},
            script_args.lang, script_args.template
        )
        dev_prompts = formatting_func(
            {"triples": [r["triples"] for r in dev], "text": [r["target"] for r in dev]},
            script_args.lang, script_args.template
        )

    tok = AutoTokenizer.from_pretrained(script_args.base_model, use_fast=False)
    # If tokenizer has no pad token, set it to eos_token (recommended for causal LM)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Optionally, add a [PAD] token as a special token if you want a dedicated pad token
    tok.add_special_tokens({"pad_token": "[PAD]"})

    def tokenize(batch):
        tokens = tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=script_args.max_seq_length
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    train_tok = Dataset.from_dict({"text": train_prompts}).map(tokenize, batched=True, remove_columns=["text"])
    dev_tok = Dataset.from_dict({"text": dev_prompts}).map(tokenize, batched=True, remove_columns=["text"])

    bnb_cfg = BitsAndBytesConfig(load_in_4bit=script_args.use_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype="auto"
    )
    model.resize_token_embeddings(len(tok))

    # --- PEFT selection ---
    if script_args.peft_type == "lora":
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(model, lora_cfg)
    elif script_args.peft_type == "prompt":
        lang_str = "Irish" if script_args.lang == "ga" else "English"
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=script_args.num_virtual_tokens,
            prompt_tuning_init_text=f"Write the following triples as fluent {lang_str} text:",
            tokenizer_name_or_path=script_args.base_model,
        )
        model = get_peft_model(model, peft_config)
    elif script_args.peft_type == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=script_args.num_virtual_tokens,
            prefix_projection=script_args.prefix_projection
        )
        model = get_peft_model(model, peft_config)
    else:
        raise ValueError(f"Unknown peft_type: {script_args.peft_type}")

    targs = TrainingArguments(
        output_dir=script_args.outdir,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        fp16=torch.cuda.is_available(),
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        logging_steps=script_args.logging_steps,
        report_to="none",
        max_grad_norm=script_args.max_grad_norm,
        weight_decay=script_args.weight_decay,
        warmup_ratio=script_args.warmup_ratio,
        eval_strategy="steps",
        eval_steps=script_args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)
    )
    trainer.train()

    model.save_pretrained(script_args.outdir)
    tok.save_pretrained(script_args.outdir)
    print(f"[finetune] Adapter saved → {script_args.outdir}")
    print("To use the best checkpoint at step 6000, see checkpoints in output dir.")

if __name__ == "__main__":
    main()
