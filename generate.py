import os
import json
from data.load_data import extract_mtriples
from utilities.llm_model import UnifiedModel, model_name
from utilities.prompt import MODEL_PROMPT, MODEL_PROMPT2, INPUT_PROMPT


# === Config ===
xml_path = "data/long-inputs_GREC_3_10.xml"
output_path = "result/output.json"
provider = "openai"

# === Load triples ===
result = extract_mtriples(xml_path)

# === Load existing progress (if any) ===
if os.path.exists(output_path):
    with open(output_path, "r") as f:
        try:
            completed_data = json.load(f)
        except json.JSONDecodeError:
            print("Warning: Output file is corrupted. Starting fresh.")
            completed_data = []
else:
    completed_data = []

# Get already processed indices
completed_indices = {entry["index"] for entry in completed_data}

# === Initialize LLM ===
conf = model_name.get(provider.lower())
llm = UnifiedModel(provider=provider, **conf).model_(MODEL_PROMPT)

# === Process Entries ===
for i, triples in enumerate(result):
    if i in completed_indices:
        continue  # skip already completed

    print(f"\nProcessing entry {i+1}/{len(result)}:")
    try:
        output = llm.invoke(INPUT_PROMPT.format(triples=triples)).content.strip()
    except Exception as e:
        print(f"Failed at index {i}: {e}")
        output = "[ERROR]"

    # Append result
    entry = {
        "index": i,
        "triples": triples,
        "output": output
    }
    completed_data.append(entry)

    # Save incrementally
    with open(output_path, "w") as f:
        json.dump(completed_data, f, indent=2)