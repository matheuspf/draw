import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/QwQ-32B-AWQ"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """I will provide a list of short image descriptions. Following that, you will generate a list of 100 similar, but distinct descriptions.

Keep some variety while following the style and size of the original descriptions.

Here are the original descriptions:

- distant orange desert at morning
- red and white t-shirt
- a pearlescent cylinder adorned with scarlet weaving
- a lighthouse overlooking a green lake
- purple rectangle on a white background
- mint parallelograms and ruby loops
- emerald tetrahedron encased in silver lattice
- snowy mountain range
- charcoal wool trousers with side stripes and hidden zipper


Note how they change in both size and style. Now you will just write new 100 descriptions similar, but distinct from the original ones. Like the ones above, just add `-` at the beginning of each line.

Start:
"""
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    temperature=0.5
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=4096
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)

