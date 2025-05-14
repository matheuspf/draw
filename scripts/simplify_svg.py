from svgpathtools import parse_path, Path, Line, CubicBezier, QuadraticBezier, Arc
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize
from lxml import etree
import re

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

# Example usage:
with open("t1.svg") as f:
    svg = f.read()
simplified_svg = simplify_svg(svg, tolerance=4, preserve_topology=True)
with open("t2.svg", "w") as f:
    f.write(simplified_svg)
