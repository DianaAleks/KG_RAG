
import os
import re
import time

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from kg_rag.utility import config_data

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
QUESTION_PATH = config_data["TRUE_FALSE_PATH"]
CACHE_DIR = config_data["LLM_CACHE_DIR"]
OUTPUT_PATH = os.path.join(
    config_data["SAVE_RESULTS_PATH"],
    "Llama_3.2_1B_Instruct_prompt_based_true_false_binary_response.csv",
)

def parse_answer(text):
    match = re.search(
        r'"answer"\s*:\s*"(true|false)"', text, re.IGNORECASE
    )
    if match:
        return match.group(1).capitalize()

    match = re.fullmatch(
        r'\s*(true|false)[.!]?\s*', text, re.IGNORECASE
    )
    if match:
        return match.group(1).capitalize()

    return "INVALID"


def main():
    start = time.time()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    questions = pd.read_csv(QUESTION_PATH)
    questions = questions.drop(columns=["Unnamed: 0"], errors="ignore")

    print(f"Loading {MODEL_ID} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()

    results = []
    if os.path.exists(OUTPUT_PATH):
        print("Output file already exists; refusing to overwrite it.")
        print("Rename or remove it only if you intentionally want a fresh run.")
        return

    for i, row in questions.iterrows():
        question = str(row["text"])
        messages = [
            {
                "role": "system",
                "content": (
                    'You are an expert biomedical researcher. '
                    'Answer the question with only this JSON format: '
                    '{"answer": "True"} or {"answer": "False"}. '
                    'Do not provide a disease identifier as the answer.'
                ),
            },
            {"role": "user", "content": question},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        raw = tokenizer.decode(
            output[0][input_length:],
            skip_special_tokens=True,
        ).strip()

        results.append({
            "question": question,
            "label": str(row["label"]),
            "llm_answer": raw,
            "parsed_answer": parse_answer(raw),
        })

        # Save progress every 10 questions
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
            print(f"Completed {len(results)}/{len(questions)}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_PATH, index=False)

    correct = (
        result_df["label"].str.strip().str.lower()
        == result_df["parsed_answer"].str.lower()
    ).sum()
    invalid = (result_df["parsed_answer"] == "INVALID").sum()

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Questions: {len(result_df)}")
    print(f"Correct: {correct}/{len(result_df)}")
    print(f"Accuracy (invalid counts as incorrect): {correct / len(result_df):.3f}")
    print(f"Invalid answers: {invalid}")
    print(f"Elapsed minutes: {(time.time() - start) / 60:.1f}")


if __name__ == "__main__":
    main()
