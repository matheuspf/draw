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
from google import genai
from google.genai import types

from google import genai
import Levenshtein
from groq import Groq

np.random.seed(42)


client = Groq(
    # This is the default and can be omitted
    api_key=os.environ.get("GROQ_API_KEY"),
)






SYSTEM_PROMPT = """You are a helpful assistant that generates questions, choices and answers for a given image caption."""


PROMPT_TEMPLATE = """Given an image caption, generate multiple-choice questions that verify if the image matches that caption.

I will provide you with a list of examples of captions and the questions and answers for each.

Strictly follow the format below. I will parse it so your response must absolutely match this format, including line breaks, commas, semicolons, etc.

Include AT LEAST 6 (six) questions. Always include at least 2 (two) negative questions. A few questions should be multiple-choice questions.

Follow the examples below:

```
Description: a purple forest at dusk
Category: landscape
Q1: What is the main setting of the image?
Choices: beach, desert, forest, mountain
A: forest
Q2: Is there anything purple in the image?
Choices: no, yes
A: yes
Q3: Is it morning in the image?
Choices: yes, no
A: no
Q4: What time of day is suggested in the image?
Choices: dawn, dusk, midday, midnight
A: dusk
Q5: What color is prominently featured in the image?
Choices: green, orange, purple, white
A: purple
Q6: Is red the main color in the image?
Choices: yes, no
A: no

Description: magenta trapezoids layered on a transluscent silver sheet
Category: geometry
Q1: Is the color silver present in the image?
Choices: yes, no
A: yes
Q2: Which word describes the silver sheet's ability to let light through?
Choices: opaque, reflective, solid, translucent
A: translucent
Q3: Are the trapezoids blue?
Choices: yes, no
A: no
Q4: Are the trapezoids layered on something?
Choices: yes, no
A: yes
Q5: Is the silver sheet opaque?
Choices: yes, no
A: no
Q6: What shape are the magenta objects?
Choices: circles, stars, trapezoids, triangles
A: trapezoids

Description: a snowy plain
Category: landscape
Q1: What covers the plain?
Choices: grass, sand, snow, water
A: snow
Q2: What is the main geographical feature depicted?
Choices: forest, mountain, ocean, plain
A: plain
Q3: Is the plain covered in lava?
Choices: yes, no
A: no
Q4: Is the plain snowy?
Choices: yes, no
A: yes
Q5: Is this a hot place?
Choices: yes, no
A: no
Q6: What is frozen in the image?
Choices: grass, sand, snow, water
A: snow

Description: gray wool coat with a faux fur collar
Category: fashion
Q1: What color is the coat?
Choices: blue, brown, gray, red
A: gray
Q2: What part of the coat has faux fur?
Choices: collar, hem, pockets, sleeves
A: collar
Q3: Are these summer clothes?
Choices: yes, no
A: no
Q4: Is the coat purple?
Choices: yes, no
A: no
Q5: What material is the coat made of?
Choices: cotton, leather, silk, wool
A: wool
Q6: Is this a warm piece of clothing?
Choices: yes, no
A: yes

Description: a bowl of creamy tomato soup with basil garnish
Category: food
Q1: What is the main dish described?
Choices: beef stew, clam chowder, creamy tomato soup, mushroom bisque
A: creamy tomato soup
Q2: What herb is used as a garnish on the soup?
Choices: chives, cilantro, basil, oregano
A: basil
Q3: What is the consistency of the soup?
Choices: brothy, chunky, creamy, watery
A: creamy
Q4: Is the soup blue in color?
Choices: yes, no
A: no
Q5: Is the soup served on a flat plate?
Choices: yes, no
A: no
Q6: Is the soup garnished with croutons?
Choices: yes, no
A: no

Description: blue flames erupting from a futuristic rocket engine
Category: rockets
Q1: What color are the flames from the engine?
Choices: green, orange, red, blue
A: blue
Q2: What is the source of the erupting flames?
Choices: dragon, flare, rocket engine, volcano
A: rocket engine
Q3: How is the style of the rocket engine described?
Choices: ancient, conventional, futuristic, steampunk
A: futuristic
Q4: Are the flames described as yellow?
Choices: yes, no
A: no
Q5: Is the engine part of an automobile?
Choices: yes, no
A: no
Q6: Is the engine described as old-fashioned?
Choices: yes, no
A: no

Description: a spinning Ferris wheel with tiny seats
Category: toys
Q1: What specific amusement ride toy is mentioned?
Choices: carousel, roller coaster, Ferris wheel, swing ride
A: Ferris wheel
Q2: What action is the Ferris wheel performing?
Choices: collapsing, launching, spinning, stopping
A: spinning
Q3: How are the seats on the Ferris wheel described?
Choices: enormous, plush, tiny, wide
A: tiny
Q4: Are the seats described as spacious?
Choices: yes, no
A: no
Q5: Is the Ferris wheel stationary?
Choices: yes, no
A: no
Q6: Is the toy a model car?
Choices: yes, no
A: no
```

Include AT LEAST 6 (six) questions. Always include at least 2 (two) negative questions. A few questions should be multiple-choice questions.

Now begin!

Description: {description}
Category: {category}
"""

# def groq_completion(prompt, system_prompt=SYSTEM_PROMPT, model="llama-3.3-70b-versatile", max_tokens=4096, temperature=0.5):
# def groq_completion(prompt, system_prompt=SYSTEM_PROMPT, model="qwen-qwq-32b", max_tokens=4096, temperature=0.5):
def groq_completion(prompt, system_prompt=SYSTEM_PROMPT, model="meta-llama/llama-4-maverick-17b-128e-instruct", max_tokens=4096, temperature=0.5):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    output = chat_completion.choices[0].message.content
    return output


def openai_completion(prompt, system_prompt=SYSTEM_PROMPT, engine="gpt-4o", max_tokens=4096, temperature=0.5):
    client = openai.OpenAI()
    
    resp =  client.chat.completions.create(
        model=engine,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        )
    
    return resp.choices[0].message.content


# def gemini_completion(prompt, model="gemini-2.5-pro-preview-03-25", max_tokens=4096, temperature=0.5, system_prompt=SYSTEM_PROMPT):
def gemini_completion(prompt, model="gemini-2.5-flash-preview-04-17", max_tokens=4096, temperature=0.5, system_prompt=SYSTEM_PROMPT):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    try_count = 0
    while try_count < 10:
        try_count += 1

        try:
            response = client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
                contents=prompt
            )
        
            if len(response.text.strip()) > 0:
                return response.text
            
            print("Gemini failed to generate content")

        except Exception as e:
            print(f"Error: {e}")
            continue
    
    raise Exception("Failed to generate content")



def parse_resp(resp):
    resp = resp.split('\n')
    
    question_instances = []
    
    this_question = None
    this_choices = None
    this_answer = None

    line_number_start = min([idx for idx, line in enumerate(resp) if line.startswith('Q1:')])
    question_number = 1

    for line_number in range(line_number_start, len(resp)):
        line = resp[line_number]
        if line.startswith(f'Q{question_number}: '):
            this_question = line[len(f'Q{question_number}: '):]
        elif line.startswith(f'Choices: '):
            this_choices = line[len(f'Choices: '):].split(', ')
        elif line.startswith(f'A: '):
            this_answer = line[len(f'A: '):]
            
            if this_question and this_choices:
                question_instances.append((this_question, this_choices, this_answer))

            this_question = None
            this_choices = None
            this_answer = None

            question_number += 1
            
    return question_instances



def postprocess_choice_answer_text(text):
    text = text.strip()
    text = text.replace('\n', '')
    return text

def postprocess_choices(choices, answer):
    choices = [postprocess_choice_answer_text(choice) for choice in choices]
    answer = postprocess_choice_answer_text(answer)
    np.random.shuffle(choices)

    if answer not in choices:
        return None, None
        
        # dists = [Levenshtein.distance(answer, choice) for choice in choices]
        # closest_choice = choices[np.argmin(dists)]
        # print(f"Answer {answer} not in choices {choices}. Using {closest_choice} instead.")
        # answer = closest_choice

    return choices, answer


def get_question_and_answers(category, caption):
    prompt = PROMPT_TEMPLATE.format(category=category, description=description)
    
    # resp = openai_completion(this_prompt)
    resp = gemini_completion(prompt)
    # resp = groq_completion(prompt)

    resp = resp.replace('```', '').strip()
    
    question_instances = parse_resp(resp)

    this_caption_qas = []
    
    for (question, choices, answer) in question_instances:
        this_qa = {}
        this_qa['category'] = category
        this_qa['caption'] = caption
        this_qa['question'] = question
        this_qa['choices'] = choices
        this_qa['answer'] = answer
        this_qa['choices'], this_qa['answer'] = postprocess_choices(this_qa['choices'], this_qa['answer'])

        this_caption_qas.append(this_qa)
        
    return this_caption_qas



if __name__ == "__main__":
    max_questions = 6
    descriptions_path = Path("/home/mpf/code/kaggle/draw/data/descriptions_gemini_2.json")
    out_path = Path("/home/mpf/code/kaggle/draw/data/questions_gemini_25_flash.parquet")
    # out_path = Path("/home/mpf/code/kaggle/draw/data/questions_groq_llama4_maverick.parquet")
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

    for category, descriptions in tqdm(descriptions.items(), total=len(descriptions)):
        for description in tqdm(descriptions, total=len(descriptions)):
            try_count = 0

            while try_count < 5:
                try_count += 1
                try:
                    qas = get_question_and_answers(category, description)

                    if len(qas) < max_questions:
                        print(f"Failed to generate {max_questions} questions for {description}. Total generated: {len(qas)}")
                        continue

                except Exception as e:
                    continue
                
                break

            if len(qas) < max_questions:
                print(f"Failed to generate {max_questions} questions for {description}. Total generated: {len(qas)}")
                print("Skipping...")
                continue

            qas = qas[:max_questions]

            dataset["category"].append(category)
            dataset["description"].append(description)
            dataset["question"].append(repr([qa["question"] for qa in qas]))
            dataset["choices"].append(repr([qa["choices"] for qa in qas]))
            dataset["answer"].append(repr([qa["answer"] for qa in qas]))
            
        pd.DataFrame(dataset).to_parquet(out_path)
