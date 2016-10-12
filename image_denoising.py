'''  
Grayscale image denoising, gibbs sampling for Ising Model
Author: Yuting Gui
Implemented algo of KPM machine learing: sec24.2 Gibbs Sampling
'''

from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from skimage.util import random_noise
from skimage.color import rgb2gray
import numpy as np
import random
import math
from optparse import OptionParser

def noise_image(name, width, length, noise_level):
    '''
    Input: image path, resize width, resize length, add noise level
    Output: the noise image, save to noise.jpg for further comparision
    '''
    # image is a rgb image
    image = imread(name)

    #convert it to gary scale
    image = rgb2gray(image)
    
    # resize to 100 * 100
    resize_image = resize(image,(width,length))
    
    # generate noise_image, save and check
    noise_image = random_noise(resize_image, mode = "s&p", amount = noise_level)
    imsave("noise.jpg", noise_image)
    return noise_image

def sigmoid(x):
    '''
    Input: a float number
    Output: a float percentage
    '''
      return 1 / (1 + math.exp(-x))

def neighbor_eta(tuple_index, x_t, image):
    '''
    Input: one pixel's index, the current pixel value, and the whole image ndarray
    generate eta_t = x_t(agree_pixel_num - disagree_pixel_num), 
    x_t shoule be -1 or 1, because of the nature of Ising Model
    Output: eta_t
    '''
    row_lower = max(0, tuple_index[0] - 1)
    row_upper = min(tuple_index[0] + 2, image.shape[0])
    column_lower = max(0, tuple_index[1] - 1)
    column_upper = min(tuple_index[1] + 2, image.shape[0])
    total = (column_upper - column_lower) * (row_upper - row_lower) -1 
    nbr = np.sum(image[row_lower:row_upper, column_lower:column_upper]) - x_t
    if int(x_t) == 1:
        eta_t = nbr - (total - nbr)
    elif int(x_t) == 0:
        eta_t = -1 * (total - nbr - nbr)
    return eta_t

def get_new_image(J, image): 
    '''
    Input: J: caupling strength
    for each pixel, based on its neighbor's value and ising prior, calculate prob for each pixel
    Output: newimage
    '''
    new_pic = np.zeros((image.shape))            
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            if image[(i,j)] < 0.5:
                image[(i,j)] = 0.0
            else:
                image[(i,j)] = 1.0

    sum_1 = np.sum(image)
    sum_0 = image.shape[0] * image.shape[0] - sum_1
    
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):

            if int(image[(i,j)]) == 1:
                local_prior = np.log(np.random.normal(loc = sum_1, scale = 2)) - np.log(np.random.normal(loc = sum_0, scale = 2))
            else:
                local_prior = np.log(np.random.normal(loc = sum_0, scale = 2)) - np.log(np.random.normal(loc = sum_1, scale = 2))
            if np.isnan(local_prior):
                local_prior = 0
            prob = sigmoid(2 * J * neighbor_eta((i,j),image[(i,j)], image) - local_prior)
            if random.random() < prob:
                new_pic[(i,j)] = 1.0
            else:
                new_pic[(i,j)] = 0.0
    return new_pic



if __name__ == "__main__":

    parser = OptionParser(usage = "usage: %prog [options]",
            version = "Gibbs Sampling for Grayscale Image Denoise, Yuting Gui",
            epilog = ''' image is the path of the target grayscale image. The program will generate a noise image and then denoise it. ''')
    parser.add_option("-p", help="image path to use", type="string", 
                      dest="imagepath")
    parser.add_option('-n', "--noise_level", metavar="level",
                      help="add how much percent of the noise to original image",
                      type="float", default=0.3, dest="noise_level")
    parser.add_option('-w', "--width", metavar="width", 
                      help="resize the original image to width (default=100)",
                      type="int", default=100, dest="width")
    parser.add_option('-l', "--length", metavar="length", 
                      help="resize the original image to length (default=100)",
                      type="int", default=100, dest="length")
    parser.add_option("-i", "--iterations", metavar="max", 
                      help="number of iterations (default=5)", 
                      type="int", default=5, dest="max_iterations")
    parser.add_option("-f", metavar="k", default=1, type="int", 
                      help="output denoise image frequence (default=1)",
                      dest="out_freq")

    (options, args) = parser.parse_args()
    if options.imagepath is None:
        parser.print_help()
        exit()

    image = noise_image(options.imagepath, options.width, options.length, options.noise_level)
    iter = 0
    while iter < options.max_iterations:
        
        this_image = get_new_image(1, image)
        if iter % options.out_freq == 0: 
            imsave(str(iter) + ".jpg", this_image)  
        image = this_image
        iter += 1


