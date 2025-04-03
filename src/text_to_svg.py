import os
import sys
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen


def load_font(font_path):
    """Load a font file and return font object with its glyph set."""
    try:
        font = TTFont(font_path)
        glyph_set = font.getGlyphSet()
        return font, glyph_set
    except Exception as e:
        print(f"Error loading font: {e}")
        return None, None


def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def split_text_into_lines(text, max_chars_per_line=None):
    """Split text into lines based on max characters per line."""
    if not max_chars_per_line:
        return [text]

    lines = []
    current_line = ""
    words = text.split()

    for word in words:
        # Check if adding this word would exceed the line length
        if len(current_line) + len(word) + (1 if current_line else 0) <= max_chars_per_line:
            # Add space before word if not the first word on the line
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            # Line would be too long, finish current line and start a new one
            lines.append(current_line)
            current_line = word

    # Don't forget to add the last line
    if current_line:
        lines.append(current_line)

    return lines


def calculate_scale_factor(font, font_size):
    """Calculate the scale factor based on font units per em."""
    units_per_em = font["head"].unitsPerEm
    return font_size / units_per_em


def render_glyph(glyph, glyph_set, scale, x_position, y_position):
    """Render a single glyph and return its path data and advance width."""
    # Get advance width
    if hasattr(glyph, "width"):
        advance_width = glyph.width * scale
    else:
        # Fallback if width not directly available
        advance_width = scale * 0.6 * 1000  # Approximate

    # Create a pen that will transform the coordinates
    svg_pen = SVGPathPen(glyph_set)
    
    # Create a transform matrix: [scale, 0, 0, -scale, x_position, y_position]
    # This scales and flips the y-axis, then translates
    transform = [scale, 0, 0, -scale, x_position, y_position]
    transform_pen = TransformPen(svg_pen, transform)

    # Draw the glyph to the transform pen
    glyph.draw(transform_pen)

    # Get the path data (already transformed)
    path_data = svg_pen.getCommands()

    return path_data, advance_width


def render_line(line_idx, svg_count, line, font, glyph_set, scale, x_position, y_position, color=(0, 0, 0)):
    """Render a single line of text and return SVG path elements."""
    svg_paths = []
    current_x = x_position

    for char_idx, char in enumerate(line):
        # Get character name in the font
        glyph_name = font.getBestCmap().get(ord(char))
        if glyph_name is None:
            continue  # Skip if character not in font

        # Get glyph object
        glyph = glyph_set[glyph_name]

        # Render the glyph
        path_data, advance_width = render_glyph(glyph, glyph_set, scale, current_x, y_position)

        if path_data:
            # Create SVG path element without transform
            path_element = f'  <path id="text-path-{svg_count + char_idx}" d="{path_data}" fill="{rgb_to_hex(*color)}" />'
            svg_paths.append(path_element)

        # Move to next character position
        current_x += advance_width

    return svg_paths


def create_svg_document(paths, width=1000, height=300):
    """Create an SVG document from a list of path elements."""
    svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'
    # Add a background rectangle
    # svg_content += f'  <rect width="{width}" height="{height}" fill="{rgb_to_hex(255, 255, 255)}" />\n'
    svg_content += "\n".join(paths)
    svg_content += "\n</svg>"
    return svg_content


def text_to_svg(
    text: str,
    svg_width: int = 384,
    svg_height: int = 384,
    # font_size: int = 40,
    x_position_frac: float = 0.1,
    y_position_frac: float = 0.7,
    line_spacing: float = 1.2,
    font_path: str = "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    color: tuple[int, int, int] = (255, 255, 255),
    font_size: int = None
):
    """Convert text to SVG paths using the specified font."""
    # Load the font file
    font, glyph_set = load_font(font_path)
    if not font or not glyph_set:
        return None
    
    # font_size = max(int(min(len(text) / 5, 50)), 20)
    
    font_size = font_size or max(min(70, 30 / (len(text) / 50)), 20)
    
    x_position = x_position_frac * svg_width
    y_position = y_position_frac * svg_height

    # Calculate scale factor
    scale = calculate_scale_factor(font, font_size)

    # Estimate average character width (approximation)
    avg_char_width = font_size * 0.5  # Approximate average character width
    
    # Calculate available width
    available_width = svg_width - x_position
    
    # Calculate max characters that can fit in the available width
    max_chars_per_line = int(available_width / avg_char_width)
    
    # Split text into lines
    lines = split_text_into_lines(text, max_chars_per_line)

    # # Calculate SVG height based on number of lines
    # svg_height = svg_height if len(lines) <= 1 else int(len(lines) * font_size * line_spacing) + 100

    # Render all lines
    all_paths = []
    current_y = y_position

    for line_idx, line in enumerate(lines):
        # Render the line
        line_paths = render_line(line_idx, len(all_paths), line, font, glyph_set, scale, x_position, current_y, color)
        all_paths.extend(line_paths)

        # Move to next line
        current_y += font_size * line_spacing

    # Create the SVG document
    svg_content = create_svg_document(all_paths, width=svg_width, height=svg_height)

    return svg_content


def save_svg(svg_content, output_file):
    """Save SVG content to a file."""
    try:
        with open(output_file, "w") as f:
            f.write(svg_content)
        return True
    except Exception as e:
        print(f"Error saving SVG: {e}")
        return False


if __name__ == "__main__":
    text = "purple pyramids spiraling around a bronze cone"
    svg = text_to_svg(text)

    with open("text_as_paths.svg", "w") as f:
        f.write(svg)
