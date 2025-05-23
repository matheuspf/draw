from PIL import Image
import subprocess
from pathlib import Path


def get_svg_primitives(
    image: str | Path | Image.Image,
    mode: int = 0,
    num_shapes: int = 50,
    temp_path: str = "/tmp/image.png",
) -> str:
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    image.save(temp_path)

    args = [
        "/path/to/primitive",
        "-i",
        temp_path,
        "-o",
        "output.svg",
        "-n",
        str(num_shapes),
        "-m",
        str(mode),
    ]
    subprocess.run(args)

    with open("output.svg", "r") as f:
        svg = f.read()

    return svg
