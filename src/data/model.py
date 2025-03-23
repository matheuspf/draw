import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

