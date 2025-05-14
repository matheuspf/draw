from PIL import Image
import io
import cairosvg

def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def gen_svg(canvas_width: int = 384, canvas_height: int = 384):
    svg = f'<svg version="2.0" viewBox="0 0 {canvas_width} {canvas_height}" xmlns="http://www.w3.org/2000/svg">\n'
    svg += f'<path d="M 0,0 h {canvas_width} v {canvas_height} h {-canvas_width} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'

    w, h = 100, 100
    svg += f'<g clip-path="rect(0, 0, {h}, {w})">\n'
    svg += f'<path d="M 0,0 h {2*w} v {2*h} h {-2*w} z" fill="{rgb_to_hex(255, 0, 0)}" />\n'
    svg += '</g>\n'
    svg += '</svg>\n'

    return svg


svg = gen_svg()

with open("output.svg", "w") as f:
    f.write(svg)

png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
img = Image.open(io.BytesIO(png_data)).convert('RGB')
img.save("output.png")
