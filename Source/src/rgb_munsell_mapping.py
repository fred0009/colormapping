import os
import pandas as pd
from PIL import Image, ImageTk
import PIL
import tkinter as tk
import numpy as np
import time
import cv2
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter

def most_frequent(x):
    central = x[x.size//2]
    values, counts = np.unique(x, return_counts=True)
    max_freq = counts.max()
    modes = values[counts == max_freq]
    if central in modes:
        return central
    else:
        return modes[0]

def cartoonize2(image, scale):
    """
    convert image into cartoon-like image
    image: input PIL image
    """

    output = np.array(image)
    x, y, c = output.shape
    hists = []
    for t in range(10):
        for i in range(c):
            output[:, :, i] = cv2.bilateralFilter(output[:, :, i], 5, 5, 50)
            # output[:, :, i] = cv2.medianBlur(output[:, :, i],3)

    output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

    hists = []
    x_hists = []
    #H
    hist, _ = np.histogram(output[:, :, 0], range=(0,180), bins=scale)
    hists.append(hist)
    x_hists.append(_)
    #S
    hist, _ = np.histogram(output[:, :, 1], range=(0,255), bins=scale)
    hists.append(hist)
    x_hists.append(_)
    #V
    hist, _ = np.histogram(output[:, :, 2], range=(0,255), bins=scale)
    hists.append(hist)
    x_hists.append(_)


    C = []
    for h in hists:
        C.append(k_histogram(h, scale))

    print("centroids: {0}".format(C))

    # Scale back histogram
    for i in range(len(C)):
        if i == 0:
            C[i] = C[i] * round(181/scale)
        else:
            C[i] = C[i] * round(256/scale)

    print("centroids: {0}".format(C))

    x, y, c = output.shape
    fig, ax = plt.subplots(3)
    for i in range(c):
        hist, x_hist = np.histogram(output[:, :, i], range=(0, 256), bins=128)
        ax[i].plot(hist, label=i)
    plt.show()
    plt.close()

    output = output.reshape((-1, c))
    for i in range(c):
        channel = output[:, i]
        index = np.argmin(np.abs(channel[:, np.newaxis] - C[i]), axis=1)
        output[:, i] = C[i][index]
    output = output.reshape((x, y, c))

    x, y, c = output.shape
    fig, ax = plt.subplots(3)
    for i in range(c):
        hist, x_hist = np.histogram(output[:, :, i], range=(0, 256), bins=128)
        ax[i].plot(hist, label=i)
    plt.show()
    plt.close()
    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)



    return output

def cartoonize(image, scale):
    """
    convert image into cartoon-like image
    image: input PIL image
    """
    output = np.array(image)
    x, y, c = output.shape

    for i in range(c):
        output[:, :, i] = cv2.bilateralFilter(output[:, :, i], 3, 500, 50)
        # output[:, :, i] = cv2.medianBlur(output[:, :, i],3)

    output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)


    peaks = []

    fig, ax = plt.subplots(3)
    m = int(scale)
    for i in range(c):
        # output[:, :, i] = cv2.bilateralFilter(output[:, :, i], 9, 350, 30)
        output[:, :, i] = output[:, :, i]
        hist, x_hist = np.histogram(output[:, :, i], bins=np.arange(180+1 if i==0 else 256+1))

        step = (x_hist[1] - x_hist[0]) * 0
        g = [ (x_hist[0] + step, hist[0]) ] if hist[0] > hist[1] else []
        g.extend( [ (x_hist[i]+step, hist[i])  for i in range(1, len(hist) - 1) if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] ])
        g.extend([ (x_hist[-1]-step,hist[-1]) ] if hist[-1] > hist[-2] else [])

        sorted_g = sorted(g, key=lambda x: x[1])
        print(i, g)
        sorted_g = [i[0] for i in sorted_g][-m:]
        print(i, sorted_g)


        peaks.append( sorted_g )
        ax[i].plot(hist, label=i)
    plt.show()

    fig, ax = plt.subplots(3)
    output = output.reshape((-1, c))
    for i in range(c):
        index = [np.argmin(np.abs(np.array(peaks[i]) - chroma)) for chroma in output[:,i] ]
        output[:, i] = np.array(peaks[i])[index]

        hist, x_hist = np.histogram(output[:, i], range=(0,256), bins=128)
        ax[i].plot(hist, label=i)
    plt.show()

    output = output.reshape((x, y, c))
    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)


    return output


def update_C(Cx, hist):
    """
    update centroids until they don't change
    """
    C = Cx
    while True:
        groups = defaultdict(list)
        #assign pixel values
        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            d = np.abs(C-i)
            index = np.argmin(d)
            groups[index].append(i)

        new_C = np.array(C)
        for i, indice in groups.items():
            if np.sum(hist[indice]) == 0:
                continue
            new_C[i] = int(np.sum(indice*hist[indice])/np.sum(hist[indice]))
        if np.sum(new_C-C) == 0:
            break
        C = new_C
    return C, groups


def k_histogram(hist, scale):
    """
    choose the best K for k-means and get the centroids
    """
    alpha = 0.005            # p-value threshold for normaltest
    N = 10 + scale                      # minimun group size for normaltest
    C = np.array([128])

    while True:
        C, groups = update_C(C, hist)

        #start increase K if possible
        new_C = set()     # use set to avoid same value when seperating centroid
        for i, indice in groups.items():
            #if there are not enough values in the group, do not seperate
            if len(indice) < N:
                new_C.add(C[i])
                continue

            # judge whether we should seperate the centroid
            # by testing if the values of the group is under a
            # normal distribution
            z, pval = stats.normaltest(hist[indice])
            if pval < alpha:
                #not a normal dist, separate
                left = 0 if i == 0 else C[i-1]
                right = len(hist)-1 if i == len(C)-1 else C[i+1]
                delta = right-left
                if delta >= 3:
                    c1 = (C[i]+left)/2
                    c2 = (C[i]+right)/2
                    new_C.add(c1)
                    new_C.add(c2)
                else:
                    # though it is not a normal dist, we have no
                    # extra space to seperate
                    new_C.add(C[i])
            else:
                # normal dist, no need to seperate
                new_C.add(C[i])
        if len(new_C) == len(C):
            break
        else:
            C = np.array(sorted(new_C))
    return C

def read_munsell_data():
    BASE_DIR = os.path.abspath('.')[:-3]

    # Read table
    df = pd.read_csv(BASE_DIR+'/database/munsell_data.csv')

    # Create counter for each row
    df['color_counts'] = [0 for x in df.iterrows()]

    return df

# Generate Test Image
def generate_test_image():
    filename = '/Users/freddy/Nanyang Technological University/Projects and Research/ColorMappingProjects/ColorMapping/tempimages/public/0f6803f78c88cb715fd7b32a056d0b8529f8bfa2_rgb.png'
    im = Image.open(filename)
    return im

# Convert to Munsell Color System
def convert_rgb_to_munsell(im, df_real):
    width, height = im.size
    pix = im.load()
    df = df_real.copy()

    for x in range(width):
        for y in range(height):
            r = pix[x,y][0]
            g = pix[x,y][1]
            b = pix[x,y][2]
            results = df.iloc[( (df['r'] - r)**2 + (df['g'] - g)**2 +
                                (df['b'] - b) ** 2).argsort()[:1]]
            hue = results['hue'].values[0]
            value = results['value'].values[0]
            chroma = results['chroma'].values[0]
            df['color_counts'][ (df['hue']==hue) & (df['value']==value) & (df['chroma']==chroma) ] += 1

    return df

def new_window(root):
    new_win = tk.Toplevel(root)
    new_win.config(bg='white')
    new_win.geometry("750x850")
    return new_win

def k_means_clustering(image, scale):
    img = np.array(image)
    # img[:, :, :] = cv2.bilateralFilter(img[:, :, :], 7, 3, 50)
    # img[:, :, :] = cv2.GaussianBlur(img[:, :, :],(19,19),0)
    img = cv2.blur(img, (5, 5))

    vectorized = img.reshape((-1, 3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = scale
    attempts=10
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))

    x, y, c = result_image.shape
    fig, ax = plt.subplots(3)
    for i in range(c):
        hist, x_hist = np.histogram(result_image[:, :, i], range=(0,256), bins=128)
        ax[i].plot(hist, label=i)
    plt.show()
    plt.close()

    return result_image

def image_vectorization(_rgb_image, scale):
    width, height = _rgb_image.size
    rgb_image = _rgb_image


    # Convert from PIL to OpenCV
    numpy_image = np.array(rgb_image)
    # opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2HSV)
    # opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    # output = cartoonize(opencv_image, scale)
    output = cartoonize2(opencv_image, scale)
    # output = k_means_clustering(opencv_image, scale)


    # Convert back from OpenCV to PIL
    pil_image = Image.fromarray( cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    # pil_image = Image.fromarray( output)

    scalex = 5
    pixels = pil_image.load()
    for i in range(width):
        for j in range(height):
            pixels[i,j] = (round(pixels[i,j][0]/scalex)*scalex, round(pixels[i,j][1]/scalex)*scalex,
                                round(pixels[i, j][2] / scalex) * scalex)
    return pixels, pil_image

def get_unique(d, colours):
    du = set()
    rgb_hvc_list = []
    for el,col in zip(d,colours):
        hvc = '({},{},{})'.format(el[0],el[1],el[2])
        rgb = (col[0],col[1],col[2])
        du.add( (el[0],el[1],el[2]) )
        rgb_hvc_list.append( (rgb,hvc) )
    return du, rgb_hvc_list

def get_unique_max(d, du):
    new_unique_max_items = []
    for unique_item in du:
        same_item = [ x[3] for x in d if (x[0],x[1],x[2]) == unique_item ]
        hvc = '({},{},{})'.format(unique_item[0],unique_item[1],unique_item[2])
        new_unique_max_items.append( {'hvc':hvc, 'count':sum(same_item)} )
    return  new_unique_max_items

def perform_mapping(image, threshold_level=5):
    df = read_munsell_data()

    rgb_image = image.convert('RGB')
    scale = threshold_level * 10
    pixels,rgb_image = image_vectorization(rgb_image, scale)

    # Make into Numpy array
    na = np.array(rgb_image)

    # Obtain each unique RGB color in the image and calculate its total counts
    colours, counts = np.unique(na.reshape(-1, 3), axis=0, return_counts=True)

    # Get the nearest HVC color for each RGB
    hvc_colours = [df.iloc[((df['r'] - r) ** 2 + (df['g'] - g) ** 2 +
                           (df['b'] - b) ** 2).argsort()[:1]] for (r,g,b) in colours]

    # Get HVC and its count, also get the list of RGB-HVC relationship for image highlight
    d = [(x['hue'].values[0],x['value'].values[0],x['chroma'].values[0],count) for x,count in zip(hvc_colours,counts)]
    d_u, rgb_hvc_list = get_unique(d,colours)
    unique_max_items = get_unique_max(d, d_u)

    # Put the information in dataframe
    for row in unique_max_items:
        df.loc[df['HVC']==row['hvc'], 'color_counts'] = row['count']

    return  rgb_image, df, rgb_hvc_list