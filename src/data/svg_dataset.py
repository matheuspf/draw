from datasets import load_dataset
import matplotlib.pyplot as plt
import os
import io
import pandas as pd
import numpy as np
from PIL import Image
import cairosvg
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import textwrap
import json
from pathlib import Path

# Create output directory if it doesn't exist
output_dir = Path("debug_output")
output_dir.mkdir(exist_ok=True)

# Load dataset
dataset = load_dataset("starvector/text2svg-stack")

def svg_to_png(svg_string, width=300, height=300):
    """Convert SVG string to PNG image using CairoSVG."""
    try:
        png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'),
                                    output_width=width,
                                    output_height=height)
        return Image.open(io.BytesIO(png_data))
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")
        # Return a blank image if conversion fails
        return Image.new('RGB', (width, height), color='white')

def wrap_text(text, width=60):
    """Wrap text to specified width."""
    return "\n".join(textwrap.wrap(text, width=width))

def visualize_sample(sample, index=None, save_dir=output_dir):
    """Visualize and save an SVG sample with its captions."""
    # Create figure with 2 subplots (1 for image, 1 for captions)
    fig = Figure(figsize=(12, 10))
    canvas = FigureCanvas(fig)
    
    # Add title with filename
    title = sample['Filename']
    if index is not None:
        title = f"Sample #{index}: {title}"
    fig.suptitle(title, fontsize=16)
    
    # First subplot for the SVG
    ax1 = fig.add_subplot(2, 1, 1)
    img = svg_to_png(sample['Svg'])
    ax1.imshow(np.array(img))
    ax1.set_title("SVG Rendering")
    ax1.axis('off')
    
    # Second subplot for the captions
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.axis('off')
    
    # Format the captions
    captions = [
        f"BLIP2: {wrap_text(sample['caption_blip2'])}",
        f"CogVLM: {wrap_text(sample['caption_cogvlm'])}",
        f"LLaVA: {wrap_text(sample['caption_llava'])}"
    ]
    
    caption_text = "\n\n".join(captions)
    ax2.text(0.05, 0.95, caption_text, 
             transform=ax2.transAxes, 
             verticalalignment='top',
             fontfamily='monospace',
             fontsize=10)
    
    # Save the figure
    filename = f"svg_sample_{index if index is not None else 'example'}.png"
    output_path = save_dir / filename
    fig.tight_layout()
    canvas.print_figure(output_path, dpi=100)
    
    # Also save raw data as JSON for reference
    json_data = {k: v for k, v in sample.items() if k != 'Svg'}
    json_data['SvgLength'] = len(sample['Svg'])
    with open(save_dir / f"{filename.replace('.png', '.json')}", 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return output_path

def explore_dataset(num_samples=5, start_idx=0, save_dir=output_dir):
    """Explore multiple samples from the dataset."""
    results = []
    
    for i in range(start_idx, start_idx + num_samples):
        if i >= len(dataset["train"]):
            break
        sample = dataset["train"][i]
        path = visualize_sample(sample, index=i, save_dir=save_dir)
        results.append(path)
        
    return results

def analyze_captions(num_samples=100, save_dir=output_dir):
    """Analyze caption statistics across models."""
    caption_lengths = {
        'blip2': [],
        'cogvlm': [],
        'llava': []
    }
    
    for i in range(min(num_samples, len(dataset["train"]))):
        sample = dataset["train"][i]
        caption_lengths['blip2'].append(len(sample['caption_blip2']))
        caption_lengths['cogvlm'].append(len(sample['caption_cogvlm']))
        caption_lengths['llava'].append(len(sample['caption_llava']))
    
    # Plot caption length distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.boxplot([caption_lengths['blip2'], 
                caption_lengths['cogvlm'], 
                caption_lengths['llava']], 
               labels=['BLIP2', 'CogVLM', 'LLaVA'])
    
    ax.set_title('Caption Length Distribution by Model')
    ax.set_ylabel('Character Count')
    
    # Save the figure
    output_path = save_dir / "caption_length_analysis.png"
    plt.savefig(output_path)
    plt.close()
    
    # Save statistics as CSV
    df = pd.DataFrame(caption_lengths)
    df.to_csv(save_dir / "caption_statistics.csv", index=False)
    
    return output_path

# Example usage:
if __name__ == "__main__":
    print("Visualizing a single example...")
    # Parse the example the user provided
    example = {
        'Filename': '006c4d2328f59111694ba675e5885a13e5ae7474.svg',
        'Svg': '<svg clip-rule="evenodd" fill-rule="evenodd" stroke-linejoin="round" stroke-miterlimit="2" viewBox="0 0 73 48" xmlns="http://www.w3.org/2000/svg"><path d="m2292.86 904.001h33.14v65.605c0 9.904 0 14.856-1.68 20.186-2.12 5.82-6.71 10.408-12.53 12.528-5.33 1.68-10.28 1.68-20.18 1.68h-14.02c-14.85 0-22.28 0-30.28 2.53-8.73 3.18-15.6 10.05-18.78 18.78-2.53 8-2.53 15.43-2.53 30.28v148.41c-27.61-.01-41.96-.18-57.37-5.06-17.46-6.35-31.22-20.1-37.57-37.56-5.06-16-5.06-30.85-5.06-60.56v-24.84c0-49.52 0-74.28 8.43-100.936 10.59-29.101 33.51-52.024 62.61-62.616 25.72-8.132 49.68-8.417 95.82-8.427z" fill="#1ed8b9" transform="matrix(-.116841 -.116841 .116841 -.116841 184.924 408.903)"/><path d="m2292.86 904.001h33.14v65.605c0 9.904 0 14.856-1.68 20.186-2.12 5.82-6.71 10.408-12.53 12.528-5.33 1.68-10.28 1.68-20.18 1.68h-14.02c-14.85 0-22.28 0-30.28 2.53-8.73 3.18-15.6 10.05-18.78 18.78-2.53 8-2.53 15.43-2.53 30.28v148.41c-27.61-.01-41.96-.18-57.37-5.06-17.46-6.35-31.22-20.1-37.57-37.56-5.06-16-5.06-30.85-5.06-60.56v-24.84c0-49.52 0-74.28 8.43-100.936 10.59-29.101 33.51-52.024 62.61-62.616 25.72-8.132 49.68-8.417 95.82-8.427z" fill="#4769d5" transform="matrix(.116841 .116841 -.116841 .116841 -112.324 -361.703)"/></svg>',
        'caption_blip2': 'the logo for the company, which is blue and green ',
        'caption_cogvlm': 'The image showcases a stylized design consisting of two intertwined shapes. The primary colors used are shades of blue and turquoise. The blue shape is larger and has a wavy, curved design, while the turquoise shape is smaller and has a more angular, jagged appearance. The design is minimalistic, with clean lines and a lack of intricate details.',
        'caption_llava': 'A blue and green logo with a wave shape.'
    }
    output_path = visualize_sample(example)
    print(f"Example visualization saved to: {output_path}")
    
    print("\nVisualizing samples from the dataset...")
    paths = explore_dataset(num_samples=50)
    print(f"Dataset samples saved to: {', '.join(str(p) for p in paths)}")
    
    print("\nAnalyzing caption statistics...")
    stats_path = analyze_captions(num_samples=100)
    print(f"Caption analysis saved to: {stats_path}")
    
    print("\nDone! All visualizations saved to the 'debug_output' directory.")