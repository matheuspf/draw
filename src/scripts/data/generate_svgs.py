import kagglehub
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import json

from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
from pydantic import BaseModel

from PIL import Image
import torch
import scour.scour

import vtracer
import tempfile
import os
from picosvg.svg import SVG

import svgpathtools
from svgpathtools import document, wsvg


import svgpathtools
from svgpathtools import svg2paths2, wsvg
import tempfile
import os

from svgpathtools import parse_path, Line, CubicBezier, QuadraticBezier, Arc
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize
from lxml import etree
import re
from src.text_to_svg import text_to_svg

from concurrent.futures import ProcessPoolExecutor
from functools import partial




class VTracerConfig(BaseModel):
    mode: str = "polygon"
    hierarchical: str = "stacked"
    colormode: str = "color"
    image_size: tuple[int, int] = (384, 384)
    filter_speckle: int = 4
    color_precision: int = 6
    layer_difference: int = 16
    corner_threshold: int = 60
    length_threshold: float = 4.0
    max_iterations: int = 10
    splice_threshold: int = 45
    path_precision: int = 8
    tolerance: float = 5.0
    preserve_topology: bool = True
    simplify_poligons: bool = True
    min_bytes: int = 9500
    # min_bytes: int = 5250


def path_area(path: svgpathtools.Path) -> float:
    """Calculate the signed area of a closed path (polygon)."""
    area = 0.0
    for seg in path:
        try:
            area += 0.5 * abs(seg.start.real * seg.end.imag - seg.end.real * seg.start.imag)
        except Exception:
            pass
    return area

def remove_smallest_paths_svg(svg: str, min_bytes: int = 10000) -> str:
    # Write SVG to a temp file for svgpathtools
    with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as tmp:
        tmp.write(svg.encode('utf-8'))
        temp_path = tmp.name

    paths, attributes, svg_attributes = svg2paths2(temp_path)
    if not paths or len(paths) == 1:
        os.remove(temp_path)
        return svg

    # Calculate areas and sort paths by area descending
    indexed = list(enumerate(paths))
    areas = [(i, abs(path_area(p))) for i, p in indexed]
    areas.sort(key=lambda x: x[1], reverse=True)
    # sorted_indices = [i for i, _ in areas]
    sorted_indices = list(range(len(areas)))

    # Binary search for the minimum number of largest paths to keep
    left, right = 1, len(paths)
    best_svg = None

    while left <= right:
        mid = (left + right) // 2
        keep_indices = sorted_indices[:mid]
        keep_paths = [paths[i] for i in keep_indices]
        keep_attributes = [attributes[i] for i in keep_indices]

        # Write new SVG to temp file
        wsvg(keep_paths, attributes=keep_attributes, svg_attributes=svg_attributes, filename=temp_path)
        with open(temp_path, "r") as f:
            candidate_svg = f.read()
        candidate_svg_opt = optimize_svg(candidate_svg)
        # candidate_svg = candidate_svg.replace("<defs/>\n", "")

        if len(candidate_svg_opt.encode('utf-8')) > min_bytes:
            best_svg = candidate_svg
            right = mid - 1
        else:
            left = mid + 1

    os.remove(temp_path)
    return best_svg if best_svg is not None else svg


def displace_svg_paths(svg, x_offset=0, y_offset=0, scale=0.5) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as tmp:
        tmp.write(svg.encode("utf-8"))
        temp_path = tmp.name

    paths, attributes, svg_attributes = svgpathtools.svg2paths2(temp_path)
    displacement = complex(x_offset, y_offset)

    for i, path in enumerate(paths):
        paths[i] = path.scaled(scale).translated(displacement)

    wsvg(paths, attributes=attributes, svg_attributes=svg_attributes, filename=temp_path)

    with open(temp_path) as f:
        svg = f.read()

    os.remove(temp_path)

    svg = svg.replace("<defs/>\n", "")

    return svg


def optimize_svg_picosvg(svg):
    svg_instance = SVG.fromstring(svg)
    svg_instance.topicosvg(inplace=True)
    return svg_instance.tostring(pretty_print=False)


def optimize_svg(svg):
    options = scour.scour.parse_args([
        '--enable-viewboxing',
        '--enable-id-stripping',
        '--enable-comment-stripping',
        '--shorten-ids',
        '--indent=none',
        '--strip-xml-prolog',
        '--remove-metadata',
        '--remove-descriptive-elements',
        '--disable-embed-rasters',
        '--enable-viewboxing',
        '--create-groups',
        '--renderer-workaround',
        '--set-precision=2',
    ])

    svg = scour.scour.scourString(svg, options)
    
    svg = svg.replace('id=""', '')
    svg = svg.replace('version="1.0"', '')
    svg = svg.replace('version="1.1"', '')
    svg = svg.replace('version="2.0"', '')
    svg = svg.replace('  ', ' ')
    svg = svg.replace('>\n', '>')
    
    return svg


def svg_conversion(img, config: VTracerConfig = VTracerConfig()):
    tmp_dir = tempfile.TemporaryDirectory()
    # Open the image, resize it, and save it to the temporary directory
    resized_img = img.resize(config.image_size)
    tmp_file_path = os.path.join(tmp_dir.name, "tmp.png")
    resized_img = resized_img.convert("RGB")
    resized_img.save(tmp_file_path)
    
    svg_path = os.path.join(tmp_dir.name, "gen_svg.svg")
    vtracer.convert_image_to_svg_py(
                tmp_file_path,
                svg_path,
                colormode=config.colormode,
                hierarchical=config.hierarchical,
                mode=config.mode,
                filter_speckle=config.filter_speckle,
                color_precision=config.color_precision,
                layer_difference=config.layer_difference,
                corner_threshold=config.corner_threshold,
                length_threshold=config.length_threshold,
                max_iterations=config.max_iterations,
                splice_threshold=config.splice_threshold,
                path_precision=config.path_precision,
            )

    with open(svg_path, 'r', encoding='utf-8') as f:
        svg = f.read()
    
    svg = optimize_svg_picosvg(svg)
    
    return svg


def polygon_to_path(svg: str) -> str:
    svg = re.sub(
        r'<polygon([\w\W]+?)points=(["\'])([\.\d, -]+?)(["\'])', 
        r'<path\1d=\2M\3z\4', 
        svg
    )
    svg = re.sub(
        r'<polyline([\w\W]+?)points=(["\'])([\.\d, -]+?)(["\'])', 
        r'<path\1d=\2M\3\4', 
        svg
    )
    return svg


def path_to_coords(path):
    coords = []
    for seg in path:
        if isinstance(seg, (Line, CubicBezier, QuadraticBezier, Arc)):
            coords.append((seg.start.real, seg.start.imag))
    # Add the last point
    if path:
        coords.append((path[-1].end.real, path[-1].end.imag))
    return coords

def coords_to_path(coords, close=False):
    if not coords:
        return ""
    d = f"M{coords[0][0]} {coords[0][1]}"
    for x, y in coords[1:]:
        d += f" L{x} {y}"
    if close:
        d += " Z"
    return d

def simplify_svg(svg_string, tolerance=1.0, preserve_topology=True):
    # Parse SVG
    root = etree.fromstring(svg_string.encode())
    ns = {'svg': root.nsmap.get(None, '')}
    for elem in root.xpath('//svg:path', namespaces=ns):
        d = elem.attrib['d']
        path = parse_path(d)
        coords = path_to_coords(path)
        if not coords:
            continue
        # Detect if path is closed
        close = (coords[0] == coords[-1])
        # Use Polygon if closed, else LineString
        if close and len(coords) > 3:
            geom = Polygon(coords)
        else:
            geom = LineString(coords)
        simplified = geom.simplify(tolerance, preserve_topology=preserve_topology)
        # Convert back to path
        if simplified.is_empty:
            continue
        if hasattr(simplified, 'exterior'):
            new_coords = list(simplified.exterior.coords)
            new_d = coords_to_path(new_coords, close=True)
        else:
            new_coords = list(simplified.coords)
            new_d = coords_to_path(new_coords, close=False)
        elem.attrib['d'] = new_d
    return etree.tostring(root, pretty_print=True).decode()



def clamp_svg(svg: str, config: VTracerConfig = VTracerConfig()):
    svg = polygon_to_path(svg)

    if config.image_size[0] != 384:
        svg = displace_svg_paths(svg, x_offset=0, y_offset=0, scale=384 / config.image_size[0])

    if config.simplify_poligons:
        svg = simplify_svg(svg, tolerance=config.tolerance, preserve_topology=config.preserve_topology)

    svg = remove_smallest_paths_svg(svg, min_bytes=config.min_bytes)

    # svg += text_to_svg("O", x_position_frac=0.85, y_position_frac=0.95, font_size=60, color=(0, 0, 0), font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf").split("\n")[1]
    svg = svg.replace("</svg>", "") + "</svg>"
    svg = optimize_svg(svg)
    svg = svg.replace(
        f'<svg baseProfile="full" viewBox="0 0 {config.image_size[0]} {config.image_size[1]}" xmlns="http://www.w3.org/2000/svg">',
        '<svg viewBox="0 0 384 384" xmlns="http://www.w3.org/2000/svg">'
    )

    return svg



def process_row(row: dict, out_folder: Path, config: VTracerConfig) -> str:
    img_path = row["image_path"]
    img = Image.open(img_path)

    svg = svg_conversion(img, config)
    svg = polygon_to_path(svg)
    svg = displace_svg_paths(svg, x_offset=0, y_offset=0, scale=384 / config.image_size[0])

    out_path = Path(out_folder) / "svgs" / row["category"] / row["image_name"] / row["prompt_name"] / f'{row["sd_seed"]}.svg'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(svg)

    return str(out_path)


def generate_vtracer_svgs(
    sdxl_dataset_path = Path("/home/mpf/code/kaggle/draw/data/sdxl/sdxl_dataset.parquet"),
    out_folder = Path("/home/mpf/code/kaggle/draw/data/vtracer"),
    max_workers = 16
):
    out_folder = Path(out_folder)

    df = pd.read_parquet(sdxl_dataset_path)
    # df = df[df["sd_seed"].isin(list(range(5)))]

    svgs = []

    config = VTracerConfig()

    rows = df.to_dict(orient="records")
    process_func = partial(process_row, out_folder=out_folder, config=config)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        svgs = list(tqdm(executor.map(process_func, rows), total=len(rows)))

    df["svg_path"] = svgs
    df.to_parquet(out_folder / "vtracer_dataset.parquet")



if __name__ == "__main__":
    generate_vtracer_svgs()
