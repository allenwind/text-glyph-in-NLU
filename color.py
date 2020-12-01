import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from imagefont import ImageFontTransformer

# ANSI escape code
ansi_code_ids =  [15, 224, 217, 210, 203, 9, 160, 124, 88]
length = len(ansi_code_ids)

hexcolors = [
    "#f5f5ff",
    "#ffe0e0",
    "#ffb6b6",
    "#ff8d8d",
    "#ff6363",
    "#ff3939",
    "#ff1010",
    "#f20000",
    "#dd0000",
    "#c80000",
    "#b40000",
    "#9f0000",
    "#8a0000",
]
hex_lenght = len(hexcolors)

template = "\033[38;5;{value}m{string}\033[0m"
def print_color_text(text, ws):
    ws = np.array(ws)
    ws = (ws - np.min(ws)) / (np.max(ws) - np.min(ws)) * 0.99
    for string, w in zip(text, ws):
        vid = int(w * length)
        value = ansi_code_ids[vid]
        print(template.format(string=string, value=value), end="")
    print()

markdown_template = '<font color="{}">{}</font>'
def render_color_markdown(text, ws):
    ws = np.array(ws)
    ws = (ws - np.min(ws)) / (np.max(ws) - np.min(ws)) * 0.99
    ss = []
    for string, w in zip(text, ws):
        i = int(w * hex_lenght)
        ss.append(markdown_template.format(hexcolors[i], string))
    return "".join(ss)

tr = ImageFontTransformer()
def render_color_image(text, ws):
    ws = np.array(ws)
    ws = (ws - np.min(ws)) / (np.max(ws) - np.min(ws)) * 0.99
    images = []
    for string, w in zip(text, ws):
        i = int(w * hex_lenght)
        hexcode = hexcolors[i]
        rgb = matplotlib.colors.hex2color(hexcode)
        image = tr.transform(string)
        image = np.concatenate([image * c for c in rgb], axis=-1)
        images.append(image)
    image = np.concatenate(images, axis=-2)
    return image

if __name__ == "__main__":
    # for testing
    import string
    text = string.ascii_letters
    print_color_text(text, np.arange(len(text)))
    print(render_color_markdown(text, np.arange(len(text))))

    image = render_color_image(text, np.arange(len(text)))
    plt.imshow(image)
    plt.show()
