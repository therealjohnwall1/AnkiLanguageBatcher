import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "tencent/HY-MT1.5-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

messages = [
    {
        "role": "user",
        "content": "Translate the following segment into English, without additional explanation.\n\nCô cho biết: trước giờ tôi không đến phòng tập công cộng.",
    },
]

tokenized_chat = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
)

# Generate with recommended parameters from docs
outputs = model.generate(
    tokenized_chat.to(model.device),
    max_new_tokens=2048,
    num_beams=5,
    early_stopping=True,
    top_k=20,
    top_p=0.6,
    repetition_penalty=1.05,
    temperature=0.7,
)

# Decode output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)
