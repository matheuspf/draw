from datasets import load_dataset

dataset = load_dataset("starvector/text2svg-stack")

print(dataset["train"][0])