# First try at manually rewriting the Kuwahara function from Luca Balbi which was written
# originally in MatLab
import numpy as np
from scipy.signal import convolve2d
import time
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt



# help on convolve2d: http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html

def Kuwahara(original):
    im = original.convert('RGB')
    numpy_image = np.array(im)
    output = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    output = np.array(output)
    x, y, c = output.shape
    tmp = 25
    xy = 3
    rep = 55
    for t in range(rep):
        for i in range(c):
            # output[:, :, i] = cv2.medianBlur(output[:, :, i], xy)
            # output[:, :, i] = cv2.blur(output[:, :, i], (xy,xy))
            output[:, :, i] = cv2.bilateralFilter(output[:, :, i], tmp, 5,50)


    # pix = im.load()
    # w,h = original.size
    # for i in range(w):
    #     for j in range(h):
    #         pix[i,j] = ( 0 if pix[i,j][0] < 125 else 255, 0 if pix[i,j][1] < 125 else 255,
    #                      0 if pix[i,j][2] < 125 else 255 )
    return output

filename = '/Users/freddy/Nanyang Technological University/Projects and Research/ColorMappingProjects/ColorMapping/tempimages/public/0f6803f78c88cb715fd7b32a056d0b8529f8bfa2_rgb.png'

im = Image.open(filename)

im  = Kuwahara(im)
im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

im.show()