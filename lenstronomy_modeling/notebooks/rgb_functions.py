_author_ = 'ajshajib'

import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import interp2d
from PIL import Image

import matplotlib as mpl

# for MNRAS
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
import matplotlib.image as mpimg
import seaborn as sns
import coloripy as cp

# to change tex to Times New Roman in mpl
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.it'] = 'serif:italic'
plt.rcParams['mathtext.bf'] = 'serif:bold'
plt.rcParams['mathtext.fontset'] = 'custom'



def set_fontscale(font_scale=2., font='Times New Roman'):
    sns.set(style='ticks', context='paper', font=font, font_scale=font_scale) #2
    #sns.set_context('talk')
    sns.set_style({"xtick.direction": "in","ytick.direction": "in", "axes.linewidth": 2.0,})
    #sns.set_palette('Set2')

set_fontscale()

cmap = sns.cubehelix_palette(start=0.5, rot=-1.5, gamma=1, hue=1, light=0., dark=1., reverse=False, as_cmap=True)
msh_cmap = cp.get_msh_cmap(num_bins=501, rescale='power', power=2.5)


def colour_norm(image, weight=1):
    """
    """
    new_image = copy.deepcopy(image)
    #new_image -= np.min(new_image) - 30
    #new_image = np.arcsinh(np.arcsinh(new_image))
    #print np.min(new_image)
    #print(np.min(new_image))
    #new_image -= np.min(new_image)
    new_image[new_image < 0] = 0
    new_image/=np.max(new_image) * weight
    new_image *= 255.99
    new_image[new_image > 255.99] = 255.99
    return new_image


def double_enlarge(image):
    """
    makes the image twice the size, takes an n*n by array, returns a 2n*2n array
    """
    size = len(image[0])
    new_image = np.zeros((size*2, size*2))
    
    for i in np.arange(size):
        for j in np.arange(size):
            new_image[i*2, j*2] = image[i, j]
            new_image[i*2+1, j*2] = image[i, j]
            new_image[i*2, j*2+1] = image[i, j]
            new_image[i*2+1, j*2+1] = image[i, j]
            
    return new_image


def interp_double_enlarge(image):
    """
    makes the image twice the size, takes an n*n by array, returns a 2n*2n array
    """
    size = len(image[0])
    new_image = np.zeros((size*2, size*2))
    
    x = np.arange(size) + 0.5
    y = np.arange(size) + 0.5
    
    interpolate = interp2d(x, y, image, 
                        kind='linear', copy=True, 
                        bounds_error=False, 
                        fill_value=0.)
    
    for i in np.arange(2*size):
        for j in np.arange(2*size):
            new_image[j, i] = interpolate(float(i)/2.+0.25, float(j)/2.+0.25)
            #new_image[i*2+1, j*2] = image[i, j]
            #new_image[i*2, j*2+1] = image[i, j]
            #new_image[i*2+1, j*2+1] = image[i, j]
            
    return new_image


def pad_values(image1, image2, value=0.):
    """
    enlarges the smaller image to match the bigger image and pad zeros outside the range of smaller image
    """
    size1 = len(image1[0])
    size2 = len(image2[0])
    
    if size1 > size2:
        big_size = size1
        small_size = size2
        bigger = image1
        smaller = image2
    elif size1 < size2:
        big_size = size2
        small_size = size1
        bigger = image2
        smaller = image1
    else:
        return image1, image2
    
    new_image = np.ones((big_size, big_size)) * value
    
    diff = int((big_size - small_size) / 2)
    for i in np.arange(small_size):
        for j in np.arange(small_size):
            new_image[i+diff, j+diff] = smaller[i, j]
            
    if size1 > size2:
        return image1, new_image
    elif size1 < size2:
        return new_image, image2

    
def crop(image, x1, x2, y1, y2):
    """
    Return the cropped image at the x1, x2, y1, y2 coordinates
    """
    if x2 == -1:
        x2=image.shape[1]-1
    if y2 == -1:
        y2=image.shape[0]-1

    mask = np.zeros(image.shape)
    mask[y1:y2+1, x1:x2+1]=1
    m = mask>0

    return image[m].reshape((y2+1-y1, x2+1-x1))

    
def crop_image (image1, image2, value=0.):
    """
    crop the bigger image to match the smaller one
    """
    size1 = len(image1[0])
    size2 = len(image2[0])
    
    diff = int(np.abs(size1 - size2) / 2)
            
    if size1 > size2:
        return image1[diff:-diff, diff:-diff], image2
    elif size1 < size2:
        return image1, image2[diff:-diff, diff:-diff]
    else:
        return image1, image2

    
def scale_image(image):
    """
    rescaling of the image pixels or logarithmic or similar scale
    """
    #return np.arcsinh(image)
    depth = 6
    for _ in range(depth):
        image = np.arcsinh(image)
    return image


def add_scale_bar(ax, length=1, fontsize=18, scale_text=None):
    """
    add a scale bar in the image to show the length in 1"
    """
    y_max, y_min = ax.get_ylim()
    d_y = y_max - y_min
    x_min, x_max = ax.get_xlim()
    
    d_x = x_max - x_min
    deltaPix = 0.04 #delta_pix_list[i]
    
    if scale_text is None:
        scale_text = '{}"'.format(length)
    
    ax.plot([d_x/20.,d_x/20.+length/deltaPix], [y_max-d_y/20.,y_max-d_y/20.], '-', color='w', linewidth=3)
        
    ax.text(length/deltaPix/2+d_x/20, y_max-d_y/15, scale_text, color='white', ha='center', fontsize=fontsize)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)

    
def name_text(ax, name, fontsize=18):
    """
    add the name of the lens to top of the image
    """
    name_x, name_y = 0.05, 0.95
    ax.annotate(name, xy=(name_x, name_y), xycoords='axes fraction', fontsize=fontsize,
                horizontalalignment='left', verticalalignment='top', color='w')
    #ax.text(name_x, name_y, name_list[i], fontsize=15, color='w')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)

    
def add_image_number(ax, model_output, sep=15, rollall=None, fontsize=18):
    """
    add the image numbering to the observed image
    """
    defl_x, defl_y = model_output['lens_light_result'][1][0]['center_x'], model_output['lens_light_result'][1][0]['center_y'] 
    
    image_ra, image_dec = model_output['ps_result'][1][0]['ra_image'], model_output['ps_result'][1][0]['dec_image']
    
    data = Data(model_output['kwargs_data'][1])
    
    image_x, image_y = data.map_coord2pix(image_ra, image_dec)
    defl_x, defl_y = data.map_coord2pix(defl_x, defl_y)
    
    #sep = 15
    labels = ['A', 'B', 'C', 'D']
    if model_output['short_name'] == 'PSJ0630':
        labels = ['A', 'B', 'C', 'D']
    else:
        order = name_orders[short_lens_names.index(model_output['short_name'])]
        labels = reorder(labels, order)
    for x, y, l in izip(image_x, image_y, labels):
        sn = (y - defl_y) / np.sqrt((y-defl_y)**2 + (x-defl_x)**2)
        cs = (x - defl_x) / np.sqrt((y-defl_y)**2 + (x-defl_x)**2)
        
        lx = x + sep * cs - 5 + rollall[1][0]
        ly = y + sep * sn + rollall[0][0]
        
        ax.text(lx, ly, l, color='white', fontsize=fontsize)
        
def add_arrows(ax, fontsize=18):
    """
    add a scale bar in the image to show the length in 1"
    """
    y_max, y_min = ax.get_ylim()
    d_y = y_max - y_min
    x_min, x_max = ax.get_xlim()
    
    d_x = x_max - x_min
    deltaPix = 0.04 #delta_pix_list[i]
    #ax.arrow([d_x/20.,d_x/20.+1./deltaPix], [y_max-d_y/20.,y_max-d_y/20.], '-', color='w', linewidth=2)
    ax.arrow(x_max-d_x/20., y_max-d_y/20, -1./deltaPix, 0, head_width=8, head_length=5, fc='white', ec='white', linewidth=3)
    ax.arrow(x_max-d_x/20., y_max-d_y/20, 0, 1./deltaPix, head_width=8, head_length=5, fc='white', ec='white', linewidth=3)
    
    ax.text(x_max-d_x/20. - 1.5/deltaPix, y_max-d_y/25, "E", color='white', ha='center', fontsize=fontsize)
    ax.text(x_max-d_x/20., y_max-d_y/20 + 1.3/deltaPix, "N", color='white', ha='center', fontsize=fontsize)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    
    
def make_rgb_image(name, data, ax1, weights=[1., 1., 1.], roll=None, sep=15, 
                   rollall=None, zoom=False, ruler_length=1, fontsize=18, scale_text=None):
    """
    draws the image of the lens from the data and model
    """
    #red = double_enlarge(data[0])
    red = double_enlarge(data[0]) 
    if roll is not None:
        for r in roll:
            red = np.roll(red, shift=r[0], axis=r[1])
            
    red, green = crop_image(red, data[1])
    red, blue = crop_image(red, data[2])
    
    if rollall is not None:
        for r in rollall:
            red = np.roll(red, shift=r[0], axis=r[1])
            green = np.roll(green, shift=r[0], axis=r[1])
            blue = np.roll(blue, shift=r[0], axis=r[1])
    
    #if zoom:
    #    red = crop(red, 60, 240, 60, 240)
    #    green = crop(green, 60, 240, 60, 240)
    #    blue = crop(blue, 60, 240, 60, 240)
    
    n = len(red[0])
    rgbArray = np.zeros((n,n,3), 'uint8')
    
    rgbArray[..., 0] = colour_norm(scale_image(red), weight=weights[0]) # r
    rgbArray[..., 1] = colour_norm(scale_image(green), weight=weights[1])# g
    rgbArray[..., 2] = colour_norm(scale_image(blue), weight=weights[2]) # b
    image = Image.fromarray(rgbArray)
    
    ax1.clear()
    
    ax1.imshow(image, origin='lower')
    if zoom:
        ax1.set_xlim(60, 240)
        ax1.set_ylim(60, 240)
        
        add_scale_bar(ax1, add=True, fontsize=fontsize, scale_text=scale_text)
    else:
        add_scale_bar(ax1, length=ruler_length, fontsize=fontsize, scale_text=scale_text)
        
    name_text(ax1, name, fontsize=fontsize)
    #add_image_number(ax1, lens_model_output, sep=sep, rollall=rollall)
    ax1.axis('off')