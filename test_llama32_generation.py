import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.2-1B-Instruct"
cache_dir = None

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto",
    cache_dir=cache_dir
)

messages = [
    {
        "role": "system",
        "content": 'Reply with exactly this JSON: {"answer": "True"}'
    },
    {
        "role": "user",
        "content": "Reply with the requested JSON."
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

print("Input tokens:", inputs["input_ids"].shape[-1])

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

new_tokens = output[0, inputs["input_ids"].shape[-1]:]
print("Generated token IDs:", new_tokens.tolist())
print("Decoded answer:", repr(tokenizer.decode(
    new_tokens, skip_special_tokens=True
)))
