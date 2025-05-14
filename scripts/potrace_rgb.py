import cv2
import numpy as np
import os
import subprocess
import tempfile
import argparse
from xml.etree import ElementTree as ET
from skimage.segmentation import slic
from scipy import ndimage


def quantize_image(img, n_colors):
    """Quantize image to n_colors using k-means."""
    Z = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(img.shape)
    return quantized, centers, labels.reshape(img.shape[:2])


def quantize_image_spatial(img, n_colors, spatial_weight=0.01):
    """Quantize image using k-means with both color and spatial information."""
    h, w = img.shape[:2]
    # Create coordinate grid
    y, x = np.mgrid[0:h, 0:w]
    # Normalize coordinates to [0,1]
    x = x / w
    y = y / h
    # Scale spatial coordinates by weight factor
    x = x * spatial_weight
    y = y * spatial_weight
    # Concatenate color and spatial info
    features = np.zeros((h, w, 5))
    features[:,:,0:3] = img / 255.0  # Normalize colors to [0,1]
    features[:,:,3] = x
    features[:,:,4] = y
    # Reshape for k-means
    features_flat = features.reshape((-1, 5)).astype(np.float32)
    # Run k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(features_flat, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Extract color centers (first 3 components)
    color_centers = np.uint8(centers[:, 0:3] * 255)
    # Apply labels to create quantized image
    quantized = color_centers[labels.flatten()].reshape(img.shape)
    return quantized, color_centers, labels.reshape(img.shape[:2])


def quantize_image_slic(img, n_colors, n_segments=500, compactness=10):
    """Quantize image using SLIC superpixels followed by k-means."""
    # Generate superpixels
    segments = slic(img, n_segments=n_segments, compactness=compactness, start_label=0)
    
    # Calculate mean color for each superpixel
    superpixel_colors = np.zeros((segments.max() + 1, 3), dtype=np.float32)
    for i in range(segments.max() + 1):
        mask = segments == i
        if np.any(mask):
            superpixel_colors[i] = np.mean(img[mask], axis=0)
    
    # Cluster superpixel colors using k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    superpixel_colors_float = superpixel_colors.astype(np.float32)
    _, sp_labels, centers = cv2.kmeans(superpixel_colors_float, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Map superpixels to their cluster centers
    centers = np.uint8(centers)
    quantized = np.zeros_like(img)
    labels_map = np.zeros(img.shape[:2], dtype=np.int32)
    
    for i in range(segments.max() + 1):
        mask = segments == i
        cluster = sp_labels[i, 0]
        quantized[mask] = centers[cluster]
        labels_map[mask] = cluster
    
    return quantized, centers, labels_map


def clean_mask(mask, min_size=100):
    """Remove small connected components from binary mask."""
    # Label connected components
    labeled, num = ndimage.label(mask)
    # Measure size of each component
    sizes = np.bincount(labeled.ravel())
    # Set size of background (label 0) to 0
    if len(sizes) > 0:
        sizes[0] = 0
    # Remove small components
    mask_sizes = sizes > min_size
    # Keep only large enough components
    cleaned = mask_sizes[labeled]
    return cleaned.astype(np.uint8)


def save_pbm(mask, filename):
    """Save a binary mask as PBM (portable bitmap)."""
    # PBM expects 0 for black, 1 for white
    pbm = np.where(mask == 0, 1, 0).astype(np.uint8)  # invert: 0=white, 1=black
    with open(filename, 'wb') as f:
        f.write(b'P4\n%d %d\n' % (pbm.shape[1], pbm.shape[0]))
        # Pack bits into bytes
        for row in pbm:
            bits = np.packbits(row)
            f.write(bits.tobytes())


def run_potrace(pbm_path, svg_path, args=[]):
    """Run potrace on PBM file to produce SVG."""
    subprocess.run(['potrace', pbm_path, '-s', '-o', svg_path, *args], check=True)


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_g_block(svg_text):
    start_pos = svg_text.find('<g transform')
    end_pos = svg_text.rfind('</g>') + len('</g>')
    if start_pos == -1 or end_pos == -1:
        return None
    return svg_text[start_pos:end_pos]


def main():
    parser = argparse.ArgumentParser(description='Trace color images with potrace.')
    parser.add_argument('input', help='Input image file')
    parser.add_argument('output', help='Output SVG file')
    parser.add_argument('--colors', type=int, default=8, help='Number of colors')
    parser.add_argument('--debug', default='debug', help='Debug output folder')
    parser.add_argument('--method', choices=['kmeans', 'spatial', 'slic'], default='slic',
                       help='Quantization method: kmeans, spatial, or slic')
    parser.add_argument('--spatial-weight', type=float, default=0.05,
                       help='Weight for spatial coordinates (for spatial method)')
    parser.add_argument('--segments', type=int, default=1000,
                       help='Number of segments for SLIC (for slic method)')
    parser.add_argument('--clean-size', type=int, default=200,
                       help='Minimum size for connected components')
    parser.add_argument('--blur', type=int, default=0,
                       help='Gaussian blur kernel size (0 for no blur)')
    # Potrace simplification parameters
    parser.add_argument('--alphamax', type=float, default=1.0,
                       help='Corner threshold parameter (higher values mean more polygon simplification)')
    parser.add_argument('--opttolerance', type=float, default=0.2,
                       help='Optimization tolerance (higher values allow more simplification)')
    parser.add_argument('--no-opticurve', action='store_true',
                       help='Turn off curve optimization')
    parser.add_argument('--no-longcurve', action='store_true',
                       help='Turn off long curve optimization')
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print('Failed to load image:', args.input)
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply blur if requested
    if args.blur > 0:
        if args.blur % 2 == 0:  # Ensure kernel size is odd
            args.blur += 1
        img = cv2.GaussianBlur(img, (args.blur, args.blur), 0)

    # Choose quantization method
    if args.method == 'spatial':
        quantized, palette, label_map = quantize_image_spatial(img, args.colors, args.spatial_weight)
    elif args.method == 'slic':
        quantized, palette, label_map = quantize_image_slic(img, args.colors, args.segments)
    else:  # kmeans
        quantized, palette, label_map = quantize_image(img, args.colors)

    # Save quantized image for debug
    ensure_dir(args.debug)
    quantized_bgr = cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(args.debug, 'quantized.png'), quantized_bgr)

    g_blocks = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, color in enumerate(palette):
            mask = (label_map == idx).astype(np.uint8)
            
            # Clean up small components
            if args.clean_size > 0:
                mask = clean_mask(mask, args.clean_size)
                
            # Save mask as PNG for debug
            mask_img = (mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.debug, f'mask_{idx:02d}.png'), mask_img)
            # Save PBM for debug
            pbm_path = os.path.join(args.debug, f'mask_{idx:02d}.pbm')
            save_pbm(mask, pbm_path)
            # Also copy PBM to temp for potrace
            pbm_tmp_path = os.path.join(tmpdir, f'mask_{idx}.pbm')
            save_pbm(mask, pbm_tmp_path)
            svg_path = os.path.join(tmpdir, f'path_{idx}.svg')
            
            # Prepare potrace arguments with simplification options
            potrace_args = ['--turdsize', str(args.clean_size), '--unit', '100']
            
            # Add polygon simplification options
            potrace_args.extend(['--alphamax', str(args.alphamax)])
            potrace_args.extend(['--opttolerance', str(args.opttolerance)])
            
            # Add optimization flags if needed
            if args.no_opticurve:
                potrace_args.append('-n')
            if args.no_longcurve:
                potrace_args.append('-O0')
                
            run_potrace(pbm_tmp_path, svg_path, potrace_args)
            
            # Save the individual SVG to debug folder
            svg_debug_path = os.path.join(args.debug, f'color_{idx:02d}.svg')
            with open(svg_path, 'rb') as f_in, open(svg_debug_path, 'wb') as f_out:
                f_out.write(f_in.read())
            # Extract <g> block
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_text = f.read()
            g_block = extract_g_block(svg_text)
            if g_block:
                # Set fill color in the <g> block
                g_block = g_block.replace('fill="#000000"', f'fill="{rgb_to_hex(color)}"')
                g_blocks.append(g_block)
            else:
                print(f"Warning: No <g> block found for color {idx}")
            # Save color swatch for debug
            swatch = np.full((32, 32, 3), color, dtype=np.uint8)
            cv2.imwrite(os.path.join(args.debug, f'color_{idx:02d}.png'), cv2.cvtColor(swatch, cv2.COLOR_RGB2BGR))

    # Create final SVG
    height, width = img.shape[:2]
    svg_header = f'''<?xml version="1.0" standalone="no"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20010904//EN"\n "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">\n<svg version="1.0" xmlns="http://www.w3.org/2000/svg"\n width="{width}.000000pt" height="{height}.000000pt" viewBox="0 0 {width}.000000 {height}.000000"\n preserveAspectRatio="xMidYMid meet">\n'''
    svg_footer = '</svg>\n'
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(svg_header)
        for g in g_blocks:
            f.write(g + '\n')
        f.write(svg_footer)
    print(f"Saved SVG to {args.output}")
    print(f"Debug images written to {args.debug}/")

if __name__ == '__main__':
    main()
