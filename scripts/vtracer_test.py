import vtracer
from PIL import Image

vtracer.convert_image_to_svg_py(
    "/home/mpf/Downloads/r1.jpg",
    "/home/mpf/Downloads/r1_2.svg",
    colormode="color",
    hierarchical="cutout",
    mode="polygon",
    filter_speckle=4,
    color_precision=6,
    layer_difference=2,
    corner_threshold=60,
    length_threshold=4.0,
    max_iterations=1,
    splice_threshold=45,
    path_precision=2,
)
