import torch
from tqdm import tqdm
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import os
import time, sys
from google import genai
from google.genai import types

from google import genai


SYSTEM_PROMPT = """You are a helpful assistant that generates descriptions for image generation based on a given topic."""

CATEGORIES = [
    "landscape",
    "fashion",
    "food",
    "activity",
    "architecture",
    "tools",
    "animals",
    "plants",
    "vehicles",
    "toys",
    "signs",
    "symbols",
    "nature",
    "geometry",
    "color",
    "industry",
    "electronics",
    "planes",
    "rockets",
    "emojis",
]

PROMPT_TEMPLATE = """Your goal is to generate a list of {count} descriptions for a given topic.

I will provide a list of examples of some topic, and you will generate a new list of 50 descriptions that are similar, but distinct from the original ones.


```
landscape:
1. a purple forest at dusk
2. a starlit night over snow-covered peaks
3. green lagoon under a cloudy sky
4. a lighthouse overlooking the ocean
5. a snowy plain

geometry:
1. khaki triangles and azure crescents
2. crimson rectangles forming a chaotic grid
3. purple pyramids spiraling around a bronze cone
4. magenta trapezoids layered on a transluscent silver sheet

fashion:
1. gray wool coat with a faux fur collar
2. burgundy corduroy pants with patch pockets and silver buttons
3. orange corduroy overalls
4. a purple silk scarf with tassel trim

food:
1. a slice of ripe watermelon with black seeds
2. dark chocolate melting over fresh strawberries
3. a steaming bowl of ramen with a soft-boiled egg
4. golden french fries with a side of ketchup

architecture:
1. towering steel and glass skyscrapers at twilight
2. ancient stone archway covered in ivy
3. a minimalist concrete building with clean lines
4. red brick facade with white window frames

rockets:
1. a sleek white rocket ascending into the blue
2. sparks flying from a retro tin toy rocket
3. a multi-stage rocket separating in orbit
4. launchpad smoke billowing around a colossal rocket

color:
1. a swirl of emerald green and sapphire blue
2. single drop of crimson in clear water
3. shifting hues of a sunset orange sky
4. a stark contrast of black and white patterns

symbols:
1. a golden ankh embossed on aged papyrus
2. a celtic knot carved into dark wood
3. intertwined silver rings on a velvet cushion
4. a bold red asterisk on a white background
```


Note how I want similar descriptions to mine:

- Similar text legnth
- Overall tone
- Punctuation and uppercase
- Similar choice of adjectives

The goal is to obtain an overall similar distribution of descriptions, such that it would be very difficult or impossible to distinguish between the new ones and original ones.

Dont say anything else, just generate the descriptions. I will parse your results exactly as they are. Dont start every description with `a` or `an` - keep it varied.

You should only export descriptions one per line, starting with the number and a dot just like in the examples.

Now, generate {count} descriptions for the topic: "{topic}".

Begin:
"""


def openai_completion(prompt, system_prompt=SYSTEM_PROMPT, engine="gpt-4o", max_tokens=4096, temperature=0.5):
# def openai_completion(prompt, system_prompt=SYSTEM_PROMPT, engine="gpt-4o-mini", max_tokens=2048, temperature=0):
    client = openai.OpenAI()
    
    resp =  client.chat.completions.create(
        model=engine,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        # stop=["\n\n", "<|endoftext|>"]
        # stop=["\n\nDescription:", "<|endoftext|>"]
        )
    
    return resp.choices[0].message.content


def gemini_completion(prompt, model="gemini-2.5-pro-preview-03-25", max_tokens=4096, temperature=0.5):
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



if __name__ == "__main__":
    count = 50
    out_path = Path("/home/mpf/code/kaggle/draw/data")
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = {}

    for category in tqdm(CATEGORIES):
        try_count = 0
        while try_count < 3:
            try_count += 1

            try:
                prompt = PROMPT_TEMPLATE.format(topic=category, count=count)

                org_result = gemini_completion(prompt)
                # org_result = openai_completion(prompt)

                result = org_result.replace("```", "").strip()
                result = result[result.find("1."):]

                descriptions = [line.split(".", 1)[1].strip() for line in result.strip().split("\n")]

                if len(descriptions) != count:
                    print(f"Warning: {category} has {len(descriptions)} descriptions, expected {count}")

                    if len(descriptions) > count:
                        descriptions = descriptions[:count]
                    else:
                        print("Missing descriptions")
                        import pdb; pdb.set_trace()
                
                if len(descriptions) == count:
                    break

            except Exception as e:
                print(f"Error: {e}")
                continue
    
        if len(descriptions) != count:
            raise Exception(f"Failed to generate {count} descriptions for {category}")

        dataset[category] = descriptions

        with open(out_path / "descriptions_gemini_2.json", "w") as f:
            json.dump(dataset, f, indent=4)
