import pandas as pd
import numpy as np
import kagglehub
import json
from src.data.model import get_model_question_gen, generate_text_question_gen
import random

def load_data_drawing_with_llms():
    # Load descriptions and questions
    drawing_with_llms_path = kagglehub.competition_download('drawing-with-llms')
    descriptions_df = pd.read_csv(f'{drawing_with_llms_path}/train.csv')
    questions_df = pd.read_parquet(f'{drawing_with_llms_path}/questions.parquet')
    
    # Create a dictionary with all questions for each ID
    questions_by_id = {}
    for _, row in questions_df.iterrows():
        if row['id'] not in questions_by_id:
            questions_by_id[row['id']] = []
            
        # Convert choices string to actual list
        # choices = row['choices'].strip("[]").replace("'", "").split()
        choices = row["choices"].tolist()
        
        questions_by_id[row['id']].append({
            'question': row['question'],
            'choices': choices,
            'answer': row['answer']
        })
    
    # Combine descriptions with their questions
    combined_data = []
    for _, row in descriptions_df.iterrows():
        item_id = row['id']
        if item_id in questions_by_id:
            combined_data.append({
                'id': item_id,
                'description': row['description'],
                'questions': questions_by_id[item_id]
            })
    
    return combined_data

def create_few_shot_prompt(examples, test_description):
    prompt = \
"""Generate questions, multiple-choice options, and answers based on a description.
Format your response as JSON. At least two questions should be multiple-choice questions, instead of yes/no.

Note how all questions are related to the image, as well as the description itself.

Generate a total of 8 distinct questions for a given description.

Examples:

"""
    for example in examples:
        prompt += f"Description: {example['description']}\n"
        prompt += f"Questions: {json.dumps(example['questions'], indent=2)}\n\n"
    
    prompt += f"Now generate questions for this description:\n"
    prompt += f"Description: {test_description}\n"
    prompt += "Questions:"
    
    return prompt

def parse_generated_questions(generated_text):
    """Parse the generated text to extract questions, choices, and answers"""
    try:
        # Try to parse the response as JSON
        start_idx = generated_text.find('[')
        end_idx = generated_text.rfind(']') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = generated_text[start_idx:end_idx]
            return json.loads(json_str)
        return []
    except Exception as e:
        print(f"Error parsing generated questions: {e}")
        return []
 
 
class ModelGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.model, self.tokenizer = get_model_question_gen(model_name)
        self.system_prompt = "You are an AI assistant that generates multiple-choice questions for descriptions. Always respond with valid JSON."
        self.data = load_data_drawing_with_llms()

        few_shot_count = 3
        np.random.seed(42)
        np.random.shuffle(self.data)
        self.few_shot_examples = self.data[:few_shot_count]

    def __call__(self, prompt: str) -> str:
        full_prompt = create_few_shot_prompt(self.few_shot_examples, prompt)
        response = generate_text_question_gen(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=full_prompt,
            system_prompt=self.system_prompt
        )
        generated_questions = parse_generated_questions(response.strip())
        
        return response

def main():
    # Load data
    data = load_data_drawing_with_llms()
    
    # Split data into few-shot examples and test examples
    random.seed(42)
    # random.shuffle(data)
    
    few_shot_count = len(data) // 2
    few_shot_examples = data[:few_shot_count]
    test_examples = data[few_shot_count:]
    
    # Load model
    model, tokenizer = get_model_question_gen()
    
    # Generate questions for test examples
    system_prompt = "You are an AI assistant that generates multiple-choice questions for descriptions. Always respond with valid JSON."
    
    # Store results for CSV
    results = []
    
    for test_example in test_examples:
        prompt = create_few_shot_prompt(few_shot_examples[:3], test_example['description'])
        
        # Generate text
        response = generate_text_question_gen(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=1024,
            temperature=0.1
        )
        
        # Parse generated questions
        generated_questions = parse_generated_questions(response.strip())
        
        # Store results
        result_item = {
            'id': test_example['id'],
            "set": "test",
            'description': test_example['description'],
            'ground_truth': test_example['questions'],
            'generated': generated_questions,
            'raw_response': response.strip()
        }
        results.append(result_item)
        
        # Display results
        print(f"ID: {test_example['id']}")
        print(f"Description: {test_example['description']}")
        print("\nGround Truth Questions:")
        print(json.dumps(test_example['questions'], indent=2))
        print("\nGenerated Questions:")
        print(len(generated_questions))
        print(json.dumps(generated_questions, indent=2))
        print("\n" + "="*80 + "\n")
    
    for train_example in few_shot_examples:
        result_item = {
            'id': train_example['id'],
            "set": "train",
            'description': train_example['description'],
            'ground_truth': train_example['questions'],
            'generated': [],
            'raw_response': ""
        }
        results.append(result_item)
    
    output_df = pd.DataFrame(results)
    # output_df.to_csv('question_generation_results.csv', index=False, sep=";")

    output_df.to_parquet("question_generation_results.parquet")
    print(f"Results saved to question_generation_results.parquet")

if __name__ == "__main__":
    main()
