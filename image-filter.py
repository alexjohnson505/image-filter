
# coding: utf-8

# In[17]:

# Instagram Reverse Engineering
# Image Filter inspired by Instagram's 'EARLYBIRD' Filter
# Alex Johnson - CS 4300 Computer Graphics Assignment 3 

# Format results for IP[y] Notebook
get_ipython().magic(u'matplotlib inline')

# Import libs
import skimage
from skimage import data, exposure
from matplotlib import pyplot as plt, cm
import numpy as np
import math

# Import skdemo from file
import imp
skdemo = imp.load_source('skdemo', 'skdemo/_skdemo.py')


# In[18]:

# Import Sample Data
robot  = data.imread("examples/robot.jpg", as_grey=False, plugin=None, flatten=None)
parade = data.imread("examples/parade.jpg", as_grey=False, plugin=None, flatten=None)

# Import Sample Photographs
alex   = data.imread("examples/alex.jpg", as_grey=False, plugin=None, flatten=None)
bike   = data.imread("examples/bike.jpg", as_grey=False, plugin=None, flatten=None)
skyrim = data.imread("examples/skyrim.jpg", as_grey=False, plugin=None, flatten=None)

# Import Sample Results (From Instagram)
alex_instagram   = data.imread("examples/alex_instagram.jpg", as_grey=False, plugin=None, flatten=None)
bike_instagram   = data.imread("examples/bike_instagram.jpg", as_grey=False, plugin=None, flatten=None)
skyrim_instagram = data.imread("examples/skyrim_instagram.jpg", as_grey=False, plugin=None, flatten=None)


# In[19]:

# Create Custom Filter to Mirror "EARLYBIRD".
# Applies multiple image effects:
# - Apply Yellow Tint
# - Increase Contrast
# - Decrease Gamma (+ Brightness)
# - Apply Vignette (Faded Border)

def filter(image):
#   image = sepia(image);         # Sepia (Currently Unused)
    image = tint(image);          # Tint
    image = contrast(image, 6.8); # Contrast
    image = gamma(image, .6);     # Gamma
    image = vignette(image);      # Vignette
    return image


# In[20]:

# Render a RGB histogram using matplotlib. 
def histogram(image):

    # Iterate throught a.) Colors, b.) Channels to create lines
    for color, channel in zip('rgb', np.rollaxis(image, axis=-1)):
        counts, bin_centers = exposure.histogram(channel)
        a = plt.fill_between(bin_centers, counts, color=color, alpha=0.4)
    return a;


# In[21]:

# Show image with Histogram
def imshow_with_histogram(image):
    skdemo.imshow_with_histogram(image)


# In[22]:

# Compare 2 images Side by Side
def compare(image1, image2):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image1)

    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.show()


# In[23]:

from skimage import img_as_ubyte

# Graph an image's distrobution of grey values.
# Useful for comparing the results of a filter.
# http://scikit-image.org/docs/dev/auto_examples/applications/plot_rank_filters.html
def greyDistrobution(image):
    acc = img_as_ubyte(image)
    hist = np.histogram(acc, bins=np.arange(0, 256))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.imshow(acc, interpolation='nearest', cmap=plt.cm.gray)
    ax1.axis('off')
    
    ax2.plot(hist[1][:-1], hist[0], lw=2)
    ax2.set_title('Histogram of grey values')


# In[24]:

# Apply a Sepia color tint to the given image.
# Thanks to user 'eickenberg' from StackOverflow
# http://stackoverflow.com/questions/23802725/using-numpy-to-apply-a-sepia-effect-to-a-3d-array

def sepia(image):     

    # Convert image to float
    # (Avoids 8bit unsigned int problems)
    img = image.astype(float) / 256.
    
    # Emulate Sepia filter by 'Dampening' colors
    # Example of the math is seen below:
    # R = .393*r + .769*g + .189&b
    # G = .349*r + .686*g + .168*b
    # B = .272*r + .534*g + .131*b

    sepia_filter = np.array([[.393, .769, .189],
                             [.349, .686, .168],
                             [.272, .534, .131]])

    scale = 0.01
    
    # Apply sepia filter
    sepia_img = img.dot(sepia_filter.T * scale)

    # Rescale filter lines
    sepia_img /= sepia_img.max()
    
    # Return image formatted as a ubyte (values 0-256)
    return skimage.img_as_ubyte(sepia_img);

# Example Sepia
# compare(parade, sepia(parade))


# In[25]:

# Add a Vignette to an image (faded border)
# We do this by iterating through each pixel,
# and dampening its color values based on its
# proximity to the center pixel of the image.
# http://stackoverflow.com/questions/522563/accessing-the-index-in-python-for-loops

def vignette(image):
        
    # Store location of center pixel
    centerX = math.floor(len(image)    / 2);
    centerY = math.floor(len(image[0]) / 2);
    
    # Iterate rows
    for x, row in enumerate(image):
        
        # Iterate columns
        for y, pixel in enumerate(row):
            
            # Determine scale (rate of fade)
            scale = darkenPercent(centerX, centerY, x, y)            
            
            # Iterate colors
            for i in [0,1,2]:
                image[x][y][i] = image[x][y][i] * scale;
                
    return image;


# In[26]:

# Returns the value by which we apply a vignette.
# Input: 2 Cartesian Points, Ex: (x1, y1), (x2, y2)
# Output: A value between 0 and 1 that corresponds
#   to the distance between the two points

def darkenPercent(x1, y1, x2, y2):
    
    # Distance Formula: Distance in pixels between 2 points
    distanceToCenterPixel = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2));
    
    # Area un-effected by shading
    offset = 1;

    # Calculate float (Scale by the max distance - x1 in this case)
    result = (1 + offset - (distanceToCenterPixel / x1));

    # Ensure result is within scaling limits.
    if result < 0:
        return 0;
    elif result > 1:
        return 1;
    else:
        return result;

# Example:
# plt.imshow(vignette(parade));


# In[27]:

# Produce the average of the 2 given images.
# Note: This function is not currently used
def blendImages(img1, img2):

    # Check for bad inputs
    if len(img1) != len(img2) or len(img1[0]) != len(img1[0]):
        print "Wrong image dimensions. Please use 2 images of the same size.";
    
    # Create accumulated output
    acc = np.copy(img1)
    
    # Save height/width/colors
    xValues = acc.shape[0]
    yValues = acc.shape[1]
    colorChannels = acc.shape[2]
    
    # Iterate through pixels
    for x in range(0, xValues - 1):
        for y in range(0, yValues - 1):
            for color_channel in range(0, colorChannels - 1):
                
                # Average each pixel's color channel
                value1 = img1[x][y][color_channel] * .8;
                value2 = img2[x][y][color_channel] * .2;
                avg = value1 + value2
                acc[x][y][i] = avg
    
    return acc;

# Example:
# plt.imshow(blendImages(parade, sepia(parade)));


# In[28]:

# Apply a specific color tint to the given image.
# Note: 'EARLYBIRD' color tint: #b39f77
def tint(image):

    # Create accumulated output
    acc = np.copy(image)
    
    # Save height/width/colors
    xValues = acc.shape[0]
    yValues = acc.shape[1]
    colorChannels = acc.shape[2]
    
    for x in range(0, xValues - 1):
        for y in range(0, yValues - 1):
            
            # Scale each color channel.
            # Reduce Blue, while keeping
            # Red and Green (yellow) close
            # to their original intensities
            acc[x][y][0] = acc[x][y][0] * 1
            acc[x][y][1] = acc[x][y][1] * .9
            acc[x][y][2] = acc[x][y][2] * 0.75
    
    return acc;


# In[29]:

# Adjust the Gamma by the provided amount (0-1)
# Thanks to skimage's built-in function
def gamma(image, gamma):
    return skimage.exposure.adjust_gamma(image, gamma);


# In[30]:

# Adjust the Contrast by the provided amount
# Thanks to skimage's built-in function
def contrast(image, gain):
    return skimage.exposure.adjust_sigmoid(image, cutoff=0.5, gain=gain, inv=False);


# In[31]:

# Show outputs along with corresponding histogram. 
# Each image set contains:
# a.) Original Image
# b.) Instagram's Output
# c.) Custom Filter's Output

print "The Original Image, Instagram Filter, Custom Filter"

imshow_with_histogram(alex)
imshow_with_histogram(alex_instagram)
imshow_with_histogram(filter(alex))


# In[33]:

print "The Original Image, Instagram Filter, Custom Filter"
imshow_with_histogram(skyrim)
imshow_with_histogram(skyrim_instagram)
imshow_with_histogram(filter(skyrim))


# In[34]:

print "The Original Image, Instagram Filter, Custom Filter"
imshow_with_histogram(bike)
imshow_with_histogram(bike_instagram)
imshow_with_histogram(filter(bike))


# In[52]:

# Mapping Grey Distrobution to Analyze Results

greyDistrobution(filter(alex))
greyDistrobution(alex_instagram)

greyDistrobution(filter(skyrim))
greyDistrobution(skyrim_instagram)

