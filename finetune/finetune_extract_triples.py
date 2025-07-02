#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
finetune_extract_triples.py

Parses a (Web)NLG-style XML file and produces a JSONL file that matches the
feature schema used by the GEM/WebNLG loaders plus the PEFT training script.

Each JSON line looks like:

{
  "gem_id":        "web_nlg_en-train-0",
  "gem_parent_id": "train/Airport/1/Id1",
  "input":         ["Aarhus_Airport | cityServed | \"Aarhus, Denmark\""],
  "target":        "The Aarhus is the airport of Aarhus, Denmark.",
  "references":    [],
  "category":      "Airport",
  "webnlg_id":     "web_nlg_en-train-0"
}
"""
import argparse
import json
import pathlib
import xml.etree.ElementTree as ET
from typing import List, Dict, Iterable


def iter_entries(xml_path: pathlib.Path) -> Iterable[ET.Element]:
    """Yield every <entry> element in the XML file lazily."""
    context = ET.iterparse(xml_path, events=("end",))
    for ev, elem in context:
        if elem.tag == "entry":
            yield elem
            elem.clear()        # Free memory


def clean_triple(raw: str) -> str:
    """Remove language tags (&quot;@en) or datatype markers (^xsd:double)."""
    return (
        raw.replace("@en", "")
        .replace("@ga", "")
        .replace("^xsd:double", "")
        .strip()
    )


def build_records(
    entry: ET.Element,
    split: str,
    lang: str,
    counter: int,
) -> List[Dict]:
    """
    Produce one record per <lex> in the specified language ('' is 'en' in WebNLG).
    -- lang = "en"  →  attribute lang=='' or 'en'
    -- lang = "ga"  →  attribute lang=='ga'
    """
    category = entry.attrib["category"]
    eid = entry.attrib["eid"]

    # 1) Triples ­– always take the <modifiedtripleset>
    triples = [
        clean_triple(m.text) for m in entry.findall("./modifiedtripleset/mtriple")
    ]

    # 2) Select lexes of the right language
    lex_elems = [
        l for l in entry.findall("./lex") if (lang == "en" and l.get("lang", "") in ("", "en"))
        or (l.get("lang") == lang)
    ]

    records = []
    for pos, lex_elem in enumerate(lex_elems):
        lid = lex_elem.get("lid")
        text = lex_elem.text.strip()

        # references = all other lexes with the same language
        refs = [le.text.strip() for le in lex_elems if le is not lex_elem]

        global_id = f"web_nlg_{lang}-{split}-{counter + pos}"
        parent_id = f"{split}/{category}/{eid}/{lid}"

        records.append(
            {
                "gem_id":        global_id,
                "gem_parent_id": parent_id,
                "input":         triples,
                "target":        text,
                "references":    refs,
                "category":      category,
                "webnlg_id":     global_id,
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, type=pathlib.Path, help="Path to *.xml")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Path to *.jsonl")
    parser.add_argument("--lang",   required=True, choices=["en", "ga"])
    parser.add_argument("--split",  default="train", help="train / dev / test (used in IDs)")
    args = parser.parse_args()

    counter = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for entry in iter_entries(args.input):
            recs = build_records(entry, args.split, args.lang, counter)
            for r in recs:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            counter += len(recs)

    print(f"Wrote {counter} records to {args.output}")


if __name__ == "__main__":
    main()

# def main():
#     pa = argparse.ArgumentParser()
#     pa.add_argument("--lang", choices=["en", "ga"], default="en")
#     pa.add_argument("--train", required=True)
#     pa.add_argument("--outdir")
#     args = pa.parse_args()

#     cfg = CFG[args.lang]
#     outdir = args.outdir or cfg["out"]

#     # Build dataset → chat prompt
#     def build_prompt(rec):
#         triples = " | ".join(rec["triples"])
#         user = f"Convert the following triples into a fluent {'Irish' if args.lang=='ga' else 'English'} description:\n{triples}"
#         return f"<s>[INST] <<SYS>>\n{cfg['sys']}\n<</SYS>>\n\n{user} [/INST] {rec['text']} </s>"

#     records = [json.loads(l) for l in Path(args.train).read_text("utf-8").splitlines()]
#     ds = Dataset.from_list([{"text": build_prompt(r)} for r in records])

#     tok = AutoTokenizer.from_pretrained(cfg["base"], 
#                                         use_fast=False
#                                         )
#     tok.pad_token = tok.eos_token
#     ds_tok = ds.map(lambda ex: tok(ex["text"], 
#                                    truncation=True, 
#                                    padding="max_length", 
#                                    max_length=1024),
#                     batched=True, 
#                     remove_columns=["text"]
#                     )

#     bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)   # replaces load_in_8bit=True flag

#     model = AutoModelForCausalLM.from_pretrained(cfg["base"], 
#                                                  #load_in_8bit=True, 
#                                                  quantization_config=bnb_cfg,
#                                                  device_map="auto",
#                                                  torch_dtype="auto",
#                                                  )
#     lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, 
#                           r=8, 
#                           lora_alpha=32, 
#                           lora_dropout=0.05,
#                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
#                           )
#     model = get_peft_model(model, lora_cfg)

#     targs = TrainingArguments(output_dir=outdir, 
#                               num_train_epochs=3,
#                               per_device_train_batch_size=4, 
#                               gradient_accumulation_steps=4,
#                               learning_rate=2e-4, 
#                               fp16=torch.cuda.is_available(),
#                               save_steps=500, 
#                               save_total_limit=2, 
#                               logging_steps=25, 
#                               report_to="none"
#                               )

#     trainer = Trainer(model=model, 
#             args=targs, 
#             train_dataset=ds_tok,
#             data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    
#     trainer.train()

#     model.save_pretrained(outdir) 
#     tok.save_pretrained(outdir)
#     print(f"[finetune] LoRA adapters saved → {outdir}")

# if __name__ == "__main__":
#     main()

