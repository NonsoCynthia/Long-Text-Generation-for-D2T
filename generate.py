import os
import json
import time
import argparse
from data.load_data import extract_mtriples
from utilities.llm_model import UnifiedModel, model_name
from utilities.prompt import (
    ENGLISH_REALIZATION_PROMPT,
    TRANSLATION_PROMPT,
    INPUT_PROMPT,
    TRANS_INPUT,
    IRISH_REALIZATION_PROMPT,
)

def rate_limiter():
    time.sleep(10)

def parse_args():
    parser = argparse.ArgumentParser(description="Run data-to-text generation or translation.")
    parser.add_argument("--model_provider", required=True, help="LLM provider (e.g., openai, ollama, hf, aixplain)")
    parser.add_argument("--model", required=False, help="Optional specific model name (e.g., gpt-4.1, claude-3-5-sonnet-latest)")
    parser.add_argument("--xml_path", required=True, help="Path to input XML file")
    parser.add_argument("--task_type", choices=["generation", "translation", "irish_generation"], required=True,
                        help="Task type: 'generation' for English, 'irish_generation' for Irish, 'translation' for translating English to Irish")
    parser.add_argument("--output_path", required=True, help="Path to save predictions (.json)")
    parser.add_argument("--translate_input", required=False, help="Path to file for translation input (.json)")
    return parser.parse_args()

def run_llm(chain, text):
    msg = chain.invoke({"input": text})
    return msg.content if hasattr(msg, "content") else str(msg)


def main():
    args = parse_args()

    xml_path = args.xml_path
    output_path = args.output_path
    provider = args.model_provider
    task_type = args.task_type

    if task_type == "generation":
        prompt_template = ENGLISH_REALIZATION_PROMPT
    elif task_type == "irish_generation":
        prompt_template = IRISH_REALIZATION_PROMPT
    else:
        prompt_template = TRANSLATION_PROMPT

    conf = model_name.get(provider.lower(), {}).copy()
    if args.model:
        conf["model_name"] = args.model
    llm = UnifiedModel(provider=provider, **conf).model_(prompt_template)

    # if "lora" in args.model:
    #     llm = UnifiedModel(provider=provider, **conf).raw_model()

    if task_type in {"generation", "irish_generation"}:
        result = extract_mtriples(xml_path)

        if os.path.exists(output_path):
            try:
                with open(output_path, "r") as f:
                    completed_data = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Output file is corrupted. Starting fresh.")
                completed_data = []
        else:
            completed_data = []

        completed_indices = {entry["index"] for entry in completed_data}

        for i, triples in enumerate(result):
            if i in completed_indices:
                continue

            print(f"[{task_type.upper()}] Entry {i+1}/{len(result)}")
            try:
                flat   = " , ".join(" | ".join(t) for t in triples)
                prompt = INPUT_PROMPT.format(triples=flat)
                
                # output = llm.invoke(prompt).strip()
                output = llm.invoke({"input": prompt}).strip()             
                # output = run_llm(llm, prompt).strip()
                if "</think>" in output:
                    output = output.split("<think>")[1].strip()
                rate_limiter()
            except Exception as e:
                print(f"Failed at index {i}: {e}")
                output = "[ERROR]"

            completed_data.append({
                "index": i,
                "triples": triples,
                "output": output
            })

            with open(output_path, "w") as f:
                json.dump(completed_data, f, indent=2)

    elif task_type == "translation":
        if not args.translate_input:
            print("Missing --translate_input argument for translation task.")
            return

        try:
            with open(args.translate_input, "r") as f:
                completed_data = json.load(f)
        except FileNotFoundError:
            print(f"Translate input file not found: {args.translate_input}")
            return
        except json.JSONDecodeError:
            print(f"Translate input file is corrupted: {args.translate_input}")
            return

        for i, entry in enumerate(completed_data):
            if "translation" in entry:
                continue

            eng_text = entry.get("output", "")
            print(f"[TRANSLATION] Entry {i+1}/{len(completed_data)}")
            try:
                prompt = TRANS_INPUT.format(english_text=eng_text)
                translation = run_llm(llm, prompt).strip()
                rate_limiter()
            except Exception as e:
                print(f"Failed at index {i}: {e}")
                translation = "[ERROR]"

            entry["translation"] = translation

            with open(output_path, "w") as f:
                json.dump(completed_data, f, indent=2)

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

if __name__ == "__main__":
    main()
