from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from starvector.data.util import process_and_rasterize_svg
import torch
 
model_name = "starvector/starvector-8b-im2svg"
 
starvector = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
processor = starvector.model.processor
tokenizer = starvector.model.svg_transformer.tokenizer
 
starvector.cuda()
starvector.eval()
 
# image_pil = Image.open('assets/examples/sample-18.png')
# image_pil = Image.open('/home/mpf/Downloads/f2.jpg')
image_pil = Image.open('/home/mpf/code/kaggle/draw_bkp/C.png').convert('RGB')
 
image = processor(image_pil, return_tensors="pt")['pixel_values'].cuda()
if not image.shape[0] == 1:
    image = image.squeeze(0)
batch = {"image": image}
 
raw_svg = starvector.generate_im2svg(batch, max_length=4000)[0]
svg, raster_image = process_and_rasterize_svg(raw_svg)

with open('output.svg', 'w') as f:
    f.write(svg)

raster_image.save('output.png')
