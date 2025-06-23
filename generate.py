import os
import json
import argparse
from data.load_data import extract_mtriples
from utilities.llm_model import UnifiedModel, model_name
from utilities.prompt import MODEL_PROMPT, TRANSLATION_PROMPT, INPUT_PROMPT, TRANS_INPUT

def parse_args():
    parser = argparse.ArgumentParser(description="Run data-to-text generation or translation.")
    parser.add_argument("--model_provider", required=True, help="LLM provider (e.g., openai, ollama, hf, aixplain)")
    parser.add_argument("--xml_path", required=True, help="Path to input XML file")
    parser.add_argument("--task_type", choices=["generation", "translation"], required=True,
                        help="Task type: 'generation' for NLG, 'translation' for translating outputs")
    parser.add_argument("--output_path", required=True, help="Path to save predictions (.json)")
    return parser.parse_args()

def main():
    args = parse_args()

    xml_path = args.xml_path
    output_path = args.output_path
    provider = args.model_provider
    task_type = args.task_type

    # Load input triples
    result = extract_mtriples(xml_path)

    # Load existing output (if any)
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

    # Initialize LLM with appropriate prompt
    prompt_template = MODEL_PROMPT if task_type == "generation" else TRANSLATION_PROMPT
    conf = model_name.get(provider.lower())
    llm = UnifiedModel(provider=provider, **conf).model_(prompt_template)

    if task_type == "generation":
        for i, triples in enumerate(result):
            if i in completed_indices:
                continue

            print(f"\n[GENERATION] Entry {i+1}/{len(result)}")
            try:
                prompt = INPUT_PROMPT.format(triples=triples)
                output = llm.invoke(prompt).content.strip()
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
        # Ensure there's something to translate
        if not completed_data:
            print("No generated output found to translate. Run generation first.")
            return

        for i, entry in enumerate(completed_data):
            if "translation" in entry:
                continue

            eng_text = entry.get("output", "")
            print(f"\n[TRANSLATION] Entry {i+1}/{len(completed_data)}")
            try:
                prompt = TRANS_INPUT.format(english_text=eng_text)
                translation = llm.invoke(prompt).content.strip()
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
