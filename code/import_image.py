import Image, ImageDraw
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
import os
import sys
import string
import colorsys
import numpy as np

sys.path.append('/Users/DicksonK/anaconda/lib/python2.7/site-packages')

def open_convert_image(input_img_file, numcolors=1):
    image = Image.open(input_img_file)
    result = image.convert('P', palette=Image.ADAPTIVE, colors=numcolors)
    result.putalpha(0)
    return(result)

def get_list_color(input_img_file, numcolors=1):
    return(open_convert_image(input_img_file, numcolors).getcolors())

def get_rgb_list(input_img_file, numcolors=1):
    colors_list = get_list_color(input_img_file, numcolors)
    result = []
    for color in colors_list:
        result.append(list(color[1][0:3]))
    return(result)

def get_hls_list(rgb_list):
    result = []
    for rgb_set in rgb_list:
        result.append((colorsys.rgb_to_hls(*[x/255.0 for x in rgb_set])))
    return(result)    

print(get_rgb_list('../img/img_001_anniesfruitybunnies.jpg'))
print(get_hls_list(get_rgb_list('../img/img_001_anniesfruitybunnies.jpg')))

def gen_dominant_color_img(input_img_file, export_img_file = None, numcolors=1, swatchsize=20):

    colors = get_list_color(input_img_file, numcolors)

    # Save colors to file

    pal = Image.new('RGB', (swatchsize*numcolors, swatchsize))
    draw = ImageDraw.Draw(pal)

    posx = 0
    for count, col in colors:
        draw.rectangle([posx, 0, posx+swatchsize, swatchsize], fill=col)
        posx = posx + swatchsize

    del draw
    if export_img_file == None:
        pal.save(string.replace(input_img_file, 'jpg', 'png'), "PNG")
    else:
        pal.save(string.replace(export_img_file, 'jpg', 'png'), "PNG")

#get_colors('input_img_file.jpg', 'export_img_file.png')

counter = 1
for image in [s for s in os.listdir('../img/') if 'jpg' in s]:
    #gen_dominant_color_img('../img/' + image, '../img/dominant_color/' + image, 3)
    #print("Processing: " + str(counter))
    counter = counter + 1


import matplotlib.pyplot as plt

from skimage import data
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.io import imread

#image = img_as_ubyte(data.camera())
'''
image = img_as_ubyte(imread('../img/img_001_anniesfruitybunnies.jpg', as_grey=True))

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))

img0 = ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title('Image')
ax0.axis('off')
fig.colorbar(img0, ax=ax0)

img1 = ax1.imshow(entropy(image, disk(5)), cmap=plt.cm.jet)
ax1.set_title('Entropy')
ax1.axis('off')
fig.colorbar(img1, ax=ax1)
'''
#plt.show()



from skimage.feature import daisy
from skimage import data
import matplotlib.pyplot as plt


'''
img = imread('../img/img_001_anniesfruitybunnies.jpg', as_grey=True)
descs, descs_img = daisy(img, step=180, radius=58, rings=2, histograms=6,
                         orientations=8, visualize=True)

fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(descs_img)
descs_num = descs.shape[0] * descs.shape[1]
ax.set_title('%i DAISY descriptors extracted:' % descs_num)
plt.show()
'''


import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data
'''

PATCH_SIZE = 21

# open the camera image
image = imread('../img/img_001_anniesfruitybunnies.jpg', as_grey=True)

# select some patches from grassy areas of the image
grass_locations = [(474, 291), (440, 433), (466, 18), (462, 236)]
grass_patches = []
for loc in grass_locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image
sky_locations = [(54, 48), (21, 233), (90, 380), (195, 330)]
sky_patches = []
for loc in sky_locations:
    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
for patch in (grass_patches + sky_patches):
    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])

# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
          vmin=0, vmax=255)
for (y, x) in grass_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
        label='Grass')
ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
        label='Sky')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLVM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(grass_patches):
    ax = fig.add_subplot(3, len(grass_patches), len(grass_patches)*1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Grass %d' % (i + 1))

for i, patch in enumerate(sky_patches):
    ax = fig.add_subplot(3, len(sky_patches), len(sky_patches)*2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Sky %d' % (i + 1))


# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14)
plt.show()
'''




'''
import struct

import scipy
import scipy.misc
import scipy.cluster
from scipy.cluster.vq import kmeans,vq


import os

NUM_CLUSTERS = float(5.0)

print 'reading image'
im = Image.open('aaa.jpg')
#im = im.resize((150, 150))      # optional, to reduce time
ar = scipy.misc.fromimage(im)
shape = ar.shape
ar = ar.reshape(scipy.product(shape[:2]), shape[2])

print 'finding clusters'
#codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
codes, dist = kmeans(ar, NUM_CLUSTERS)
print 'cluster centres:\n', codes

vecs, dist = vq(ar, codes)         # assign codes
counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

index_max = scipy.argmax(counts)                    # find most frequent
peak = codes[index_max]
colour = ''.join(chr(c) for c in peak).encode('hex')
print 'most frequent is %s (#%s)' % (peak, colour)
'''