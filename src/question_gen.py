import json
import torch
import math
import numpy as np
from tqdm import tqdm
import random

np.random.seed(42)


# Landscapes
# Abstract
# Fashion
# Potential Additional Categories
# Food & Beverages - fruits, vegetables, dishes, drinks
# Architecture - buildings, structures, homes, monuments
# Furniture - chairs, tables, beds, shelves
# Household Items - kitchen utensils, appliances, decor items
# Nature Elements - plants, flowers, trees, animals (non-human)
# Transportation - vehicles, boats, aircraft (generic)
# Geometric Patterns - shapes, designs, symmetrical arrangements
# Weather Phenomena - clouds, rainbows, storms, seasons
# Tools & Equipment - hammers, wrenches, gardening tools
# Electronics - generic devices, gadgets, screens
# Musical Instruments - guitars, pianos, drums
# Sports Equipment - balls, rackets, goals, equipment
# Stationery - pens, papers, notebooks, office supplies
# Toys & Games - blocks, board games, playthings (non-human)
# Celestial Objects - stars, planets, moon, space phenomena
# Crafts & Art Supplies - yarn, paint, brushes, materials
# Industrial Objects - machinery, factory elements, infrastructure



# categories = ['object', 'human', 'animal', 'food', 'activity', 'attribute', 'counting', 'color', 'material', 'spatial', 'location', 'shape', 'other']


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



prompt = f"""
Given a image descriptions, generate four multiple-choice questions that verifies if the image description is correct. Always include a negative question.
Classify each concept into a type {categories}, and then generate a question for each type.

Strictly follow the format below. I will parse it so your response must absolutely match this format, including line breaks, commas, semicolons, etc.

Examples:

``````
Description: distant orange desert at morning
Entities: desert
Activities:
Colors: orange
Counting:
Other attributes: distant, morning
Questions and answers are below:
About desert (landscape):
Q: is this a desert?
Choices: yes, no
A: yes
Q: what type of landscape is depicted?
Choices: desert, mountain, forest, beach
A: desert
About orange (color):
Q: is the desert orange?
Choices: yes, no
A: yes
Q: what color is the desert?
Choices: orange, blue, green, purple
A: orange
About distant (spatial):
Q: is the desert distant?
Choices: yes, no
A: yes
Q: is the desert distant or close?
Choices: distant, close, overhead, underwater
A: distant
About morning (time):
Q: is it morning?
Choices: yes, no
A: yes
Q: what time of day is it?
Choices: morning, afternoon, evening, night
A: morning

Description: charcoal wool trousers with side stripes and hidden zipper
Entities: trousers, stripes, zipper
Activities:
Colors: charcoal
Counting:
Other attributes: wool, side, hidden
Questions and answers are below:
About trousers (fashion):
Q: are these trousers?
Choices: yes, no
A: yes
Q: what type of clothing item is this?
Choices: trousers, shirt, jacket, skirt
A: trousers
About stripes (attribute):
Q: do the trousers have stripes?
Choices: yes, no
A: yes
Q: what design feature do the trousers have?
Choices: stripes, polka dots, floral pattern, plain
A: stripes
About zipper (fashion):
Q: is there a zipper?
Choices: yes, no
A: yes
About charcoal (color):
Q: are the trousers charcoal colored?
Choices: yes, no
A: yes
Q: what color are the trousers?
Choices: charcoal, blue, white, brown
A: charcoal
About wool (attribute):
Q: are the trousers made of wool?
Choices: yes, no
A: yes
Q: what material are the trousers made of?
Choices: wool, cotton, polyester, leather
A: wool
About side (spatial):
Q: are the stripes on the side?
Choices: yes, no
A: yes
Q: where are the stripes located?
Choices: side, front, back, bottom
A: side
About hidden (attribute):
Q: is the zipper hidden?
Choices: yes, no
A: yes
Q: is the zipper hidden or visible?
Choices: hidden, visible, prominent, decorative
A: hidden

Description: a purple rocket landing on a green moon
Entities: rocket, moon
Activities: landing
Colors: purple, green
Counting:
Other attributes:
Questions and answers are below:
About rocket (rockets):
Q: is there a rocket?
Choices: yes, no
A: yes
Q: what spacecraft is in the image?
Choices: rocket, satellite, space station, rover
A: rocket
About moon (nature):
Q: is there a moon?
Choices: yes, no
A: yes
Q: what celestial body is in the image?
Choices: moon, planet, star, asteroid
A: moon
About landing (activity):
Q: is the rocket landing?
Choices: yes, no
A: yes
Q: what is the rocket doing?
Choices: landing, launching, orbiting, exploding
A: landing
About purple (color):
Q: is the rocket purple?
Choices: yes, no
A: yes
Q: what color is the rocket?
Choices: purple, red, blue, white
A: purple
About green (color):
Q: is the moon green?
Choices: yes, no
A: yes
Q: what color is the moon?
Choices: green, blue, white, gray
A: green

Description: a pearlescent cylinder adorned with scarlet weaving
Entities: cylinder, weaving
Activities:
Colors: pearlescent, scarlet
Counting:
Other attributes: adorned
Questions and answers are below:
About cylinder (shape):
Q: is this a cylinder?
Choices: yes, no
A: yes
Q: what shape is described?
Choices: cylinder, cube, sphere, pyramid
A: cylinder
About weaving (attribute):
Q: is there weaving on the cylinder?
Choices: yes, no
A: yes
Q: what decoration technique is on the cylinder?
Choices: weaving, painting, carving, printing
A: weaving
About pearlescent (color):
Q: is the cylinder pearlescent?
Choices: yes, no
A: yes
Q: what finish does the cylinder have?
Choices: pearlescent, matte, glossy, textured
A: pearlescent
About scarlet (color):
Q: is the weaving scarlet?
Choices: yes, no
A: yes
Q: what color is the weaving?
Choices: scarlet, blue, green, purple
A: scarlet
About adorned (attribute):
Q: is the cylinder adorned with something?
Choices: yes, no
A: yes
Q: is the cylinder plain or adorned?
Choices: adorned, plain, damaged, incomplete
A: adorned

Description: a lighthouse overlooking a green lake
Entities: lighthouse, lake
Activities: overlooking
Colors: green
Counting:
Other attributes:
Questions and answers are below:
About lighthouse (architecture):
Q: is there a lighthouse?
Choices: yes, no
A: yes
Q: what structure is in the image?
Choices: lighthouse, castle, skyscraper, bridge
A: lighthouse
About lake (nature):
Q: is there a lake?
Choices: yes, no
A: yes
Q: what body of water is shown?
Choices: lake, ocean, river, pond
A: lake
About overlooking (spatial):
Q: is the lighthouse overlooking the lake?
Choices: yes, no
A: yes
Q: what is the lighthouse doing in relation to the lake?
Choices: overlooking, floating on, submerged in, reflecting in
A: overlooking
About green (color):
Q: is the lake green?
Choices: yes, no
A: yes
Q: what color is the lake?
Choices: green, blue, brown, clear
A: green

Description: emerald tetrahedron encased in silver lattice
Entities: tetrahedron, lattice
Activities:
Colors: emerald, silver
Counting:
Other attributes: encased
Questions and answers are below:
About tetrahedron (shape):
Q: is this a tetrahedron?
Choices: yes, no
A: yes
Q: what geometric shape is described?
Choices: tetrahedron, cube, sphere, cylinder
A: tetrahedron
About lattice (abstract):
Q: is there a lattice?
Choices: yes, no
A: yes
Q: what structure surrounds the tetrahedron?
Choices: lattice, cage, box, frame
A: lattice
About emerald (color):
Q: is the tetrahedron emerald?
Choices: yes, no
A: yes
Q: what color is the tetrahedron?
Choices: emerald, ruby, sapphire, amber
A: emerald
About silver (color):
Q: is the lattice silver?
Choices: yes, no
A: yes
Q: what color is the lattice?
Choices: silver, gold, bronze, copper
A: silver
About encased (spatial):
Q: is the tetrahedron encased in the lattice?
Choices: yes, no
A: yes
Q: how is the tetrahedron positioned relative to the lattice?
Choices: encased in, on top of, next to, under
A: encased in

Description: red and white t-shirt
Entities: t-shirt
Activities:
Colors: red, white
Counting:
Other attributes:
Questions and answers are below:
About t-shirt (fashion):
Q: is this a t-shirt?
Choices: yes, no
A: yes
Q: what type of clothing is this?
Choices: t-shirt, pants, jacket, sweater
A: t-shirt
About red (color):
Q: is the t-shirt red?
Choices: yes, no
A: yes
About white (color):
Q: is the t-shirt white?
Choices: yes, no
A: yes
Q: what colors is the t-shirt?
Choices: red and white, blue and white, black and white, green and white
A: red and white
``````


Strictly follow the format above. I will parse it so your response must absolutely match this format, including line breaks, commas, semicolons, etc.

Now begin!

Description: """

def parse_resp(resp):
    resp = resp.split('\n')
    
    question_instances = []
    
    this_entity = None
    this_type = None
    this_question = None
    this_choices = None
    this_answer = None
    
    for line_number in range(6, len(resp)):
        line = resp[line_number]
        if line.startswith('About '):
            whole_line = line[len('About '):-1]
            this_entity = whole_line.split(' (')[0]
            this_type = whole_line.split(' (')[1].split(')')[0]
            
        elif line.startswith('Q: '):
            this_question = line[3:]
        elif line.startswith('Choices: '):
            this_choices = line[9:].split(', ')
        elif line.startswith('A: '):
            this_answer = line[3:]
            
            if this_entity and this_question and this_choices:
                question_instances.append((this_entity, this_question, this_choices, this_answer, this_type))
            this_question = None
            this_choices = None
            this_answer = None
            
    return question_instances


## Generate questions for a caption with GPT-3

def postprocess_choice_answer_text(text):
    text = text.strip()
    text = text.replace('\n', '')
    return text

def postprocess_choices(choices):
    choices = [postprocess_choice_answer_text(choice) for choice in choices]
    np.random.shuffle(choices)

    return choices


def palligema_completion(evaluator, prompt):
    inputs = evaluator.processor(text=prompt,return_tensors='pt').to(torch.float16).to(evaluator.model.device)
    input_len = inputs['input_ids'].shape[-1]

    with torch.inference_mode():
        outputs = evaluator.model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    outputs = outputs[0][input_len:]
    decoded = evaluator.processor.decode(outputs, skip_special_tokens=True)

    return decoded


def get_question_and_answers(evaluator, caption):
    this_prompt = prompt + caption + "\nEntities: "
    # resp = openai_completion(this_prompt)
    # resp = gemini_completion(this_prompt)
    resp = palligema_completion(evaluator, this_prompt)

    # with open('resp.json', 'w') as f:
    #     json.dump(resp, f)
    
    question_instances = parse_resp(resp)
    
    this_caption_qas = []
    
    for question_instance in question_instances:
        this_qa = {}
        this_qa['caption'] = caption
        this_qa['element'] = question_instance[0]
        this_qa['question'] = question_instance[1]
        this_qa['choices'] = question_instance[2]
        this_qa['answer'] = question_instance[3]
        this_qa['element_type'] = question_instance[4]
        
        if question_instance[4] not in categories:
            continue
            
        # if this_qa['element_type'] in ['animal', 'human']:
        #     this_qa['element_type'] = 'animal/human'
            
        this_qa['choices'] = postprocess_choices(this_qa['choices'])
        this_qa['answer'] = postprocess_choice_answer_text(this_qa['answer'])

        if this_qa['answer'] not in this_qa['choices']:
            continue
        
        this_caption_qas.append(this_qa)
        
    return this_caption_qas
