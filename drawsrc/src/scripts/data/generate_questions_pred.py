import json
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import random
import json
from pathlib import Path
import openai
import os
import time, sys

np.random.seed(42)

import pandas as pd
import numpy as np
import kagglehub
import json
import random
import torch
import vllm

from src.scripts.data.generate_questions import gemini_completion, parse_resp, postprocess_choices, postprocess_choice_answer_text


SYSTEM_PROMPT = """You are a helpful assistant that generates questions, choices and answers for a given image caption."""


categories = [
    "landscape",
    "abstract",
    "fashion",
    "food",
    "activity",
    "architecture",
    "tools",
    "animals",
    "plants",
    "vehicles",
    "attribute",
    "toys",
    "signs",
    "symbols",
    "nature",
    "spatial",
    "color",
    "shape",
    "person",
    "industry",
    "electronics",
    "planes",
    "rockets",
    "emojis",
    "other"
]


PROMPT_TEMPLATE = """Given an image caption, generate multiple-choice questions that verify if the image matches that caption.

I will provide you with a list of examples of captions and the questions and answers for each.

Strictly follow the format below. I will parse it so your response must absolutely match this format, including line breaks, commas, semicolons, etc.

Include AT LEAST 6 (six) questions. Always include at least 2 (two) negative questions. A few questions should be multiple-choice questions.

Follow the examples below:


```
Description: a green lagoon under a cloudy sky
Category: landscape
Q1: Is the lagoon depicted as green?
Choices: yes, no
A: yes
Q2: Is there a lagoon present in the image?
Choices: yes, no
A: yes
Q3: Does it look like a clear day in the image?
Choices: yes, no
A: no
Q4: What is above the lagoon?
Choices: ceiling, roof, sky, trees
A: sky
Q5: What body of water is depicted in the image?
Choices: lagoon, mountain, ocean, river
A: lagoon
Q6: Is the lagoon blue?
Choices: yes, no
A: no

Description: orange corduroy overalls
Category: fashion
Q1: What material is the item?
Choices: corduroy, denim, leather, silk
A: corduroy
Q2: Is a hat depicted?
Choices: yes, no
A: no
Q3: Is the item made of corduroy?
Choices: yes, no
A: yes
Q4: What type of clothing is shown?
Choices: a dress, a skirt, a suit, overalls
A: overalls
Q5: What color is the item?
Choices: blue, green, orange, purple
A: orange
Q6: Is this a social dress?
Choices: yes, no
A: no

Description: a maroon dodecahedron interwoven with teal threads
Category: geometry
Q1: Are the threads colored pink?
Choices: yes, no
A: no
Q2: Is there a triangle interwoven with teal threads?
Choices: yes, no
A: no
Q3: Is there anything maroon in the image?
Choices: yes, no
A: yes
Q4: Are there any teal elements in the image?
Choices: yes, no
A: yes
Q5: How many sides does the object have?
Choices: 4, 5, 6, 12
A: 12
Q6: What is the relationship between the threads and the object?
Choices: interwoven, layered, wrapped, braided
A: interwoven

Description: crispy bacon strips on a white plate
Category: food
Q1: What is the main food item depicted?
Choices: eggs, pancakes, bacon strips, sausages
A: bacon strips
Q2: What is the texture of the bacon?
Choices: chewy, crispy, raw, soft
A: crispy
Q3: Is the plate yellow?
Choices: yes, no
A: no
Q4: What are the bacon strips served on?
Choices: a bowl, a cutting board, a napkin, a white plate
A: a white plate
Q5: What color is the plate?
Choices: black, blue, red, white
A: white
Q6: Is the bacon described as burnt?
Choices: yes, no
A: no
```


Strictly follow the format above. I will parse it so your response must absolutely match this format, including line breaks, commas, semicolons, etc.

Note how questions and answers SHOULD NOT contain commas - otherwise the parsing will fail.

DO NOT forget to also output the category.

The answer must absolutely be EXACTLY one of the choices, character by character the EXACT same words.

Include AT LEAST 6 (six) questions. Always include at least 2 (two) negative questions. A few questions should be multiple-choice questions.

Now begin!

Description: {description}
"""



def get_model_question_gen(model_name: str = "qwen-lm/qwen2.5/transformers/7b-instruct-awq/1", device="cuda:0"):
    model_path = kagglehub.model_download(model_name)
    model = vllm.LLM(
        model_path,
        quantization="awq" if "awq" in model_name else "gptq",
        tensor_parallel_size=1, 
        gpu_memory_utilization=0.49, 
        trust_remote_code=True,
        dtype=torch.float16, 
        enforce_eager=True,
        max_model_len=2048,
        device=device
    )
    tokenizer = model.get_tokenizer()
    return model, tokenizer


def generate_text_question_gen(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 2560,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42
):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    outputs = model.chat(
        messages,
        vllm.SamplingParams(
            n=1,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            skip_special_tokens=False,
            max_tokens=max_new_tokens,
        ),
        use_tqdm=False
    )
    response = outputs[0].outputs[0].text

    return response

 
class QuestionGenerator:
    def __init__(self, device="cuda:0"):
        self.model, self.tokenizer = get_model_question_gen(device=device)
        self.system_prompt = SYSTEM_PROMPT

    def __call__(self, description: str) -> str:
        response = generate_text_question_gen(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=PROMPT_TEMPLATE.format(description=description),
            system_prompt=self.system_prompt,
            temperature=0.5,
        )
        
        return response


def get_question_and_answers(generator, description):
    resp = generator(description)
    resp = resp.replace('```', '').strip()

    question_instances = parse_resp(resp)
    category = resp.split("Category: ")[1].split("\n")[0]

    this_caption_qas = []
    
    for (question, choices, answer) in question_instances:
        this_qa = {}
        this_qa['category'] = category
        this_qa['caption'] = description
        this_qa['question'] = question
        this_qa['choices'] = choices
        this_qa['answer'] = answer
        this_qa['choices'], this_qa['answer'] = postprocess_choices(this_qa['choices'], this_qa['answer'])

        this_caption_qas.append(this_qa)
        
    return this_caption_qas



if __name__ == "__main__":
    descriptions_path = Path("/home/mpf/code/kaggle/draw/data/descriptions_gemini_2.json")
    out_path = Path("/home/mpf/code/kaggle/draw/data/questions_qwen25.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(descriptions_path, 'r') as f:
        descriptions = json.load(f)

    dataset = {
        "category": [],
        "description": [],
        "question": [],
        "choices": [],
        "answer": []
    }

    generator = QuestionGenerator()
    
    for category, descriptions in tqdm(descriptions.items(), total=len(descriptions)):
        max_questions = 6

        for description in tqdm(descriptions, total=len(descriptions)):

            num_tries = 10
            
            for tries in range(num_tries):
                try:
                    qas = get_question_and_answers(generator, description)

                    if len(qas) < max_questions:
                        print(f"Failed to generate {max_questions} questions for {description}. Total generated: {len(qas)}")
                        continue

                    break
                except Exception as e:
                    print(e)
                    continue
            
            if len(qas) < max_questions:
                print(f"Failed to generate {max_questions} questions for {description}. Total generated: {len(qas)}")
            
            qas = qas[:max_questions]

            dataset["category"].append(category)
            dataset["description"].append(description)
            dataset["question"].append(repr([qa["question"] for qa in qas]))
            dataset["choices"].append(repr([qa["choices"] for qa in qas]))
            dataset["answer"].append(repr([qa["answer"] for qa in qas]))
            
        pd.DataFrame(dataset).to_parquet(out_path)
