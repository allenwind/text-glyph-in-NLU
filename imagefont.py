from PIL import ImageFont
import matplotlib.pyplot as plt
import numpy as np

class ImageFontTransformer:

    def __init__(self, file="fonts/STKAITI.TTF", size=30, pad_size=32):
        self.file = file
        self.size = size
        self.pad_size = pad_size
        self.font = ImageFont.truetype(file, size=size)

    def transform(self, c):
        return self.char_to_glyph(c)

    def char_to_glyph(self, c):
        mask = self.font.getmask(c)
        size = mask.size[::-1]
        glyce = np.asarray(mask).reshape(size)
        # glyce[glyce != 0] = 255
        return self.pad_array(glyce)[:,:,np.newaxis] / 255

    def pad_array(self, array):
        # pad to center with same shape
        shape = array.shape
        a1 = (self.pad_size - shape[0]) // 2
        a2 = (self.pad_size - shape[0]) - a1

        b1 = (self.pad_size - shape[1]) // 2
        b2 = (self.pad_size - shape[1]) - b1
        array = np.pad(
            array,
            [(a1, a2), (b1, b2)],
            mode="constant",
            constant_values=0
        )
        return array

if __name__ == "__main__":
    tr = ImageFontTransformer()
    text = "NLP的魅力在于不断探索"
    cs = []
    for c in text:
        glyce = tr.transform(c)
        print(glyce.shape)
        cs.append(glyce)
    cs = np.concatenate(cs, axis=1)
    plt.imshow(cs)
    plt.show()
