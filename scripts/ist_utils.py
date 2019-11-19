
import os
import sys
import random
import itertools
import colorsys
from scripts.style_utils import save_img
from scripts.wct import WCT
import numpy as np
import skimage
from skimage.measure import find_contours
from skimage import dtype_limits
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import IPython.display
import scipy
import tensorflow as tf
from collections import Counter
from random import shuffle


def rgb2grayscale(image):
    """
    this function converts an 3-d numpy representation of the image to
    2-d numpy representation of grayscale image

    Input:
        image: 3-d np array of the image

    Output:
        gray_image: 2-d np array of the image
    """

    assert image.ndim == 3 and image.shape[2] == 3

    gray_image = np.dot(image, [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    return gray_image


def grayscale2rgb(gray_image):
	"""
	this function converts a 2-d numpy representation of a grayscale image
	to a 3-d numpy representation of grayscale image

	Input:
		gray_image: 2-d np array of the grayscale image

	Output:
		image: 3-d np array of the grayscale image
	"""

	assert gray_image.ndim == 2

	image = np.stack([gray_image, gray_image, gray_image], axis = -1).astype(np.uint8)

	return image

def mask2grayscale(image, mask):
	"""
	this function is to to convert the object covered in mask to
	grayscale

	Input:
		image: 3-d np array of the image
		mask: 2-d np array of the mask covering the object

	Note:
		the height and width of the mask must equal to the heigh and width of
		the image

	Output:
		output: the image after the masked area is turned into grayscale
	"""

	assert image.shape[:2] == mask.shape

	## to increase the number of channel of the mask to three so that we can apply the masks
	## to image
	masks = np.stack([mask, mask, mask], axis = -1)


	## the second argument turns the masked area to 2-d grayscale first and
	## turns that back to three channel grayscale
	output = np.where(masks == 1, grayscale2rgb(rgb2grayscale(image)),
		image).astype(np.uint8)

	return output


def adjust_brightness(image, mask, gamma):
	"""
	this function is to use gamma correction to adjust the brightness
	of an object of the input image.

	For more information, please refer to https://en.wikipedia.org/wiki/Gamma_correction

	This function did not include the gain parameter

	Input:
		image: a 3-d numpy representation of the image
		mask: 2-d np array of the mask covering the object
		gamma: a non-negative parameter to control the change of brightness of the object;
			   gamma > 1 will increase the brightness and gamma < 1 will decrease
			   brightness; an int

	Output:
		output: the image after the brightness of the object is adjusted
	"""

	assert image.shape[:2] == mask.shape and gamma > 0

	## to increase the number of channel of the mask to three so that we can apply the masks
	## to image
	masks = np.stack([mask, mask, mask], axis = -1)

	scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])

	output = np.where(masks == 1,
						(image / scale) ** (1 / gamma) * scale,
						image)

	## to make sure the pixel intensity is within the range of uint8
	output = np.clip(output, 0, 255).astype(np.uint8)

	return output


def extract_object(image, mask):
	"""
	This function is to extract object covered by the mask and pads this object
	with zeros so that the output image has the same shape as the input image

	Input:
		image: a 3-d numpy representation of the image
		mask: 3-d or 2-d np array of the mask covering the object

	Output:
		output:
			an image with only the object convered by the mask

	"""

	assert image.ndim == 3 and image.shape[2] == 3 and image.shape[:2] == mask.shape[:2]

	if mask.ndim == 2:
		## expand dimension so that the array is 3-d
		mask = np.expand_dims(mask, axis = -1)
		output = np.multiply(image, mask).astype(np.uint8)
	else:

		## **there may be faster way to perform this operation
		mask = np.expand_dims(mask, axis = 2)

		output = 0
		for i in range(mask.shape[-1]):
			output += np.multiply(image, mask[:, :, :, i])

		output = output.astype(np.uint8)


	return output


def interpolation(array, method):
	"""
	This function is to interpolate 1-d array

	Inputs:

		array: 1-d numpy array to be interpolated
		method: method to interpolate, "nearest neighbor" or "linear"; a string
	Output:
		array: the array after interpolation; 1-d array
	"""

	nonzero_indices = np.where(array != 0)[0]
	zero_indices = np.where(array == 0)[0]

	## linear interpolation
	if method == "linear":

		for i in range(len(nonzero_indices) - 1):

			interval = np.linspace(array[nonzero_indices[i]], array[nonzero_indices[i+1]], nonzero_indices[i+1] - nonzero_indices[i] + 1)
			array[nonzero_indices[i]: nonzero_indices[i+1] + 1] = interval

	## nearest neighbor interpolation
	elif method == "nearest neighbor":
		for i in zero_indices:
			index = np.searchsorted(nonzero_indices, i)
			if i - nonzero_indices[index - 1] < nonzero_indices[index] - i:
				array[i] = array[nonzero_indices[index - 1]]
			else:
				array[i] = array[nonzero_indices[index]]

	array = array.astype("uint8")
	return array







def get_matrix_shape(shape, padding = True):
	"""
	This function is to get the shape of the matrix into which we will
	assign the pixels. Please note that here we want the ratio between width
	and height to be as close to 1 as possible

	Input:
		paddin: to pad zeros if height / width is less than 1/5 or greater than 5; a boolean

	Outputs:
		width: the width of the matrix; an int
		height: the height of the matrix; an int

	"""


	## we will start the search of the width and height from the square root
	## of the total size
	width = np.round(shape**0.5)

	while shape % width != 0:
		width -= 1

	height = shape / width


	while padding and (height / width < 1/2 or height / width > 2):

		shape += 1

		width, height = get_matrix_shape(shape)

	assert 1/2 <= height / width <= 2, "The shape of the matrix is too irregular"

	return int(width), int(height)

## there may be better approach to this function
def image_align(first_image, second_image):
    """
    this function is to align two images when their sizes are slightly
    different from each other because of processing done to one or both
    of them. We will align the second image to the first_image

    note: the size of the second image(the first two channels) must be greater
          the size of the first image

    This function is used in the stylize function because the stylized
    image's size is changed after going through the WCT algorithm. We need
    to align this output with the original content image to "put" back
    the stylized object

    Inputs:
        first_image: 3-d np array
        second_image: 3_d np array

    Output:
        aligned_image: 3-d np array

    """

    high_diff = (second_image.shape[0] - first_image.shape[0]) // 2
    width_diff = (second_image.shape[1] - first_image.shape[1]) // 2

    align_image = second_image[high_diff: high_diff + first_image.shape[0],
                               width_diff: width_diff + first_image.shape[1],
                               :]


    assert align_image.shape == first_image.shape

    return align_image




def stylize(content_image, mask, style_image, content_image_name, content_size = 0,
			save_output = False, visualize = True, alpha  = 1.0, style_size = 512,
			crop_size = 0):
	"""
	this function is to stylize the object extracted from mask r-cnn
	with the style from the style image using the algorithm of
	Universal Style Transfer via Feature Transforms

	The code in this function borrows heavily from https://github.com/eridgd/WCT-TF/blob/master/stylize.py

	Note: for reference to the paper: https://arxiv.org/pdf/1705.08086.pdf

	Inputs:
		content_image: the content image from which we want to extract the object; 3-d np array
		mask: the mask with which we will extract object in the content_image;
		      3-d or 2-d np array of the mask covering the object
		style_image: the style image with which we apply the style; a str of url
		content_image_name: the content image name; a str
		content_size: resize the short site of the content image; an int; default is 0
		save_output: to save or not to save the final output; a boolean; default is False
		visualize: to visualize the final output or not; a boolean; default is True
		alpha: the balance between style and content; a float
		style_size: resize the short site of the content image; an int; default is 512
		crop_size: crop square size, default 256; an int

	"""

	assert content_image.ndim == 3 and mask.ndim <= 3

	cp = ["./models/relu5_1",
		  "./models/relu4_1",
	      "./models/relu3_1",
	      "./models/relu2_1",
	      "./models/relu1_1"]
	relu_targets = ["relu5_1", "relu4_1", "relu3_1", "relu2_1", "relu1_1"]
	#*****************
	## need to modify checkpoints, relu_targets, and vgg_path
	wct_model = WCT(checkpoints=cp,
	                        relu_targets=relu_targets,
	                        vgg_path='./models/vgg_normalised.t7'

	                        )



	"""
	if content_size > 0:
	        content_img = resize_to(content_imgage, content_size)
	"""

	object_image = extract_object(content_image, mask)

	for style_fullpath in style_image:
		style_prefix, style_ext = os.path.splitext(style_fullpath)
		style_prefix = os.path.basename(style_prefix)  # Extract filename prefix without ext

		style_img = skimage.io.imread(style_fullpath)

		if style_size > 0:
			style_img = resize_to(style_img, style_size)
		if crop_size > 0:
			style_img = center_crop(style_img, crop_size)

		"""
		if keep_colors:
			style_img = preserve_colors_np(style_img, content_img)
		"""
		# Run the frame through the style network

		stylized_rgb = wct_model.predict(object_image, style_img, alpha).astype("uint8")


		"""
		if passes > 1:
		    for _ in range(passes-1):
		        stylized_rgb = wct_model.predict(stylized_rgb, style_img, alpha, swap5, ss_alpha, adain)

		# Stitch the style + stylized output together, but only if there's one style image
		if concat:
		    # Resize style img to same height as frame
		    style_img_resized = scipy.misc.imresize(style_img, (stylized_rgb.shape[0], stylized_rgb.shape[0]))
		    # margin = np.ones((style_img_resized.shape[0], 10, 3)) * 255
		    stylized_rgb = np.hstack([style_img_resized, stylized_rgb])
		"""

		if mask.ndim == 2:
			## increase the dimension of the mask to 3
			mask = np.stack([mask, mask, mask], axis = -1)

		## the stylized_rgb size may not be equal to the original content image size
		stylized_rgb = image_align(content_image, stylized_rgb)



		stylized_rgb = np.where(mask == 1, stylized_rgb, 0)


		hollowed_content_image = np.where(mask == 1, 0, content_image)
		## put the stylized object back to the original content image
		stylized_rgb += hollowed_content_image

		# the final output will be saved a folder called stylized_output under current working directory
		if save_output:
			if not os.path.isdir("stylized_output"):
				os.mkdir("stylized_output")

			outpath = os.path.join(os.getcwd(), "stylized_output")

			outpath = os.path.join(outpath, '{}_{}{}'.format(content_image_name, style_prefix, style_ext))

			save_img(outpath, stylized_rgb)

		if visualize:
			## visualize the stylized output
			_, ax = plt.subplots(1, figsize = (12, 12))
			ax.imshow(stylized_rgb)

	return stylized_rgb



## This is our approach
def stylization(stretched_image, style_image,
		alpha = 1.0, style_size = 512, crop_size = 0):

	"""
	this function is to stylize the object based on the stretched method developed
	in the paper with the style from the style image using the algorithm of
	Universal Style Transfer via Feature Transforms
	The code in this function borrows heavily from https://github.com/eridgd/WCT-TF/blob/master/stylize.py
	Note: for reference to the paper: https://arxiv.org/pdf/1705.08086.pdf
	Inputs:
		stretched_image: the stretched content image; 3-d np array

		style_image: the style image with which we apply the style; a str of url

		alpha: the balance between style and content; a float
		style_size: resize the short site of the content image; an int; default is 512
		crop_size: crop square size, default 256; an int
	"""
	tf.reset_default_graph()

	assert stretched_image.ndim == 3
	
	cp = ["./models/relu5_1",
			  "./models/relu4_1",
		      "./models/relu3_1",
		      "./models/relu2_1",
		      "./models/relu1_1"]
	relu_targets = ["relu5_1", "relu4_1", "relu3_1", "relu2_1", "relu1_1"]
		#*****************
		## need to modify checkpoints, relu_targets, and vgg_path
	wct_model = WCT(checkpoints=cp,
		            relu_targets=relu_targets,
		            vgg_path='./models/vgg_normalised.t7'

		                        )


	for style_fullpath in style_image:
		style_prefix, style_ext = os.path.splitext(style_fullpath)
		style_prefix = os.path.basename(style_prefix)  # Extract filename prefix without ext

		style_img = skimage.io.imread(style_fullpath)

		if style_size > 0:
			style_img = resize_to(style_img, style_size)
		if crop_size > 0:
			style_img = center_crop(style_img, crop_size)

		"""
	    if keep_colors:
	      style_img = preserve_colors_np(style_img, content_img)
	    """
	    # Run the frame through the style network

		stylized_rgb = wct_model.predict(stretched_image, style_img, alpha).astype("uint8")


	    ## the stylized_rgb size may not be equal to the original content image size
		stylized_rgb = image_align(stretched_image, stylized_rgb)


	return stylized_rgb



def restore(stylized_obj, mask, orig_img):
	"""
	This function is to put the stylized_object back to the original
	image

	Inputs:
	stylized_obj: the stylized object; a 3-d numpy array
	mask: the corresponding mask; a 2-d numpy array
	orig_img: the original image from which the object was extracted; a 3-d numpy array

	Output:
	img: the final output in which the stylized_obj is put back; a 3-d numpy array

	"""

	assert stylized_obj.ndim == 3

	top_x_coordinate = np.sort(np.where(mask != 0)[0])[0]
	left_y_coordinate = np.sort(np.where(mask != 0)[1])[0]
	bottom_x_coordinate = np.sort(np.where(mask != 0)[0])[::-1][0]
	right_y_coordinate = np.sort(np.where(mask != 0)[1])[::-1][0]

	channels = []
	for channel in range(3):
		channels.append(np.pad(stylized_obj[:, :, channel],
											((top_x_coordinate, orig_img.shape[0] - bottom_x_coordinate), (left_y_coordinate, orig_img.shape[1] - right_y_coordinate)), "constant"))


	stylized_obj = np.stack([channels[0], channels[1], channels[2]], axis = -1)

	## to scrap the area covered by the mask
	masks = np.stack([mask, mask, mask], axis = -1)
	hollowed_img = np.where(masks == 1, 0, orig_img)

	img = hollowed_img + stylized_obj

	return img



def ss(image1, image2, hue_angel = 60, saturation_interval_size = 10, value_interval_size = 10):

	"""
	This function calculates the style similarity of two images using the
	method metioned in the paper

	Inputs:
		image1: an RGB 3-d numpy array
		image2: an RGB 3-d numpy array
		hue_angel: the degree of each hue area; an int between 0 - 360
		saturation_interval_size: the size of each interval in the saturation; an int
		value_interval_size: the size of each interval in the value/lightness; an int



	Output:
		style_sim: style similarity between two images; a float
	"""

	assert image1.shape[-1] == 3 and image2.shape[-1] == 3, "only RGB images are accpted"
	assert 1 <= saturation_interval_size <= 100, "saturation_interval_size recommended to be between 1 and 100"
	assert 1 <= value_interval_size <= 100, "value_interval_size recommended to be between 1 and 100"

	dis1, color1 = get_col_dist(image1, hue_angel, saturation_interval_size, value_interval_size)
	dis2, color2 = get_col_dist(image2, hue_angel, saturation_interval_size, value_interval_size)

	## to make sure the lengths of two distributions are the same
	if len(dis1) >= len(dis2):

		dis2 = np.pad(dis2, (0, len(dis1) - len(dis2)), "constant")
	else:
		dis1 = np.pad(dis1, (0, len(dis2) - len(dis1)), "constant")

	## the distribution difference
	dis_diff = (np.sum((dis1 - dis2) ** 2) / len(dis1)) ** 0.5

	"""
	hue_diff = get_hue_diff(color1, color2)

	saturation_diff = channel_sqrdiff(color1, color2, 2, 100 / saturation_interval_size)

	value_diff = channel_sqrdiff(color1, color2, 3, 100 / value_interval_size)

	color_difference = diff_aggregate(hue_diff, saturation_diff, value_diff,
		weights = (dis1 + dis2) / 2)

	"""
	return dis_diff#, color_difference



def get_col_dist(image, hue_angel, saturation_interval_size, value_interval_size):
	"""
	This function is to get the color distribution of the image

	Inputs:
		image: a non-normalized RGB image
		hue_angel: the degree of each hue area; an int between 0 - 360
		saturation_interval_size: the size of each interval in the saturation; an int
		value_interval_size: the size of each interval in the value/lightness; an int

	Outputs:
		dist: the color distribution
		color: a dictionary where the key is the color and the value is the
			   proportion that this color appears in this image

	"""
	image = rgb_to_hsv(image / 255.0)

	##convert the hue channel into angel degrees
	image[:, :, 0] = image[:, :, 0] * 360

	##to quantize the hue channel
	image[:, :, 0] = np.floor(image[:, :, 0] / hue_angel)

	##to quantize the saturation channel
	image[:, :, 1] = np.floor(image[:, :, 1] * 100 / saturation_interval_size)

	##to quantize the value channel
	image[:, :, 2] = np.floor(image[:, :, 2] * 100/ value_interval_size)


	color = [tuple(image[i, j, :]) for i in range(image.shape[0]) for j in range(image.shape[1])]


	## the hsv distributions for images
	dis = Counter(color)

	## normalize the distribution to account for different sizes of images
	dis = {i:dis[i]/(image.size/3) for i in dis}

	dis = dict(sorted(dis.items(), key = lambda x: x[1], reverse = True))

	color = {(i[0] * hue_angel, i[1], i[2]):dis[i] for i in dis}

	dis = np.array(list(dis.values()))

	return dis, color



def channel_sqrdiff(dict1, dict2, position, normalize_constant = 1):
	"""
	This function is specifically desighed to extract numbers in the same
	position of keys in two different dictionaries and put them into array
	and calculate the square different of these two arrays

	Note: the lengths of these two arrays must be same

	Inputs:
		dict1: the first dictionary
		dict2: the second dictionary
		position: the position in the key from which we want to extract the number
		normalize_constant: the constant to normalize the array; default to 1
	Output:
		sqr_diff: the square difference between the arrays; an array
	"""

	assert len(dict1) == len(dict2), "lengthes of the dictionaries not same"

	array1 = np.array([i[position] for i in dict1]) / normalize_constant
	array2 = np.array([i[position] for i in dict2]) / normalize_constant

	return (array1 - array2) ** 2



def get_hue_diff(dict1, dict2, normalize_constant = 180):

	"""
	This function is to specifically get the square difference of the
	hue channel because of the different way we handle the values in
	this channel

	Inputs:
		dict1: the first dictionary
		dict2: the second dictionary
		normalize_constant: the constant to normalize the array; default to 1
	Output:
		array: the square difference between the arrays; an array
	"""

	assert len(dict1) == len(dict2), "lengthes of the dictionaries not same"

	array1 = np.array([i[position] for i in dict1])
	array2 = np.array([i[position] for i in dict2])

	array = (np.minimum(np.abs(array1 - array2), 360 - np.abs(array1 - array2)) / normalize_constant) ** 2

	return array

def diff_aggregate(h, s, v, weights, h_weight = 1/3, s_weight = 1/3, v_weight = 1/3):
	"""
	This function is to aggregate the weighted square differences of the h,
	s, and v channels into what is called color difference; this metric measures
	how different the colors are in the two images

	Inputs:
		h: the square difference array of the hue channel
		s: the square difference array of the saturation channel
		v: the square difference array of the value channel
		h_weight: the weight for h; a float
		s_weight: the weight for s; a float
		v_weight: the weight for v; a float
		weights: the weight array for each of the bin

	Output:
		color_difference: the weighted color difference across the two images; a float

	"""
	assert h_weight + s_weight + v_weight == 1

	diff = (h_weight * h + s_weight * s + v_weight * v) ** 0.5

	color_difference = np.dot(diff, weight)

	return color_difference




def de_colorazation(image, hue_angel, saturation_interval_size, value_interval_size):


	image = rgb_to_hsv(image / 255.0)

	##convert the hue channel into angel degrees
	image[:, :, 0] = image[:, :, 0] * 360

	##to quantize the hue channel
	image[:, :, 0] = np.floor(image[:, :, 0] / hue_angel) * hue_angel / 360

	##to quantize the saturation channel
	image[:, :, 1] = np.floor(image[:, :, 1] * 100 / saturation_interval_size) * saturation_interval_size / 100

	##to quantize the value channel
	image[:, :, 2] = np.floor(image[:, :, 2] * 100/ value_interval_size) * value_interval_size / 100


	rgb = hsv_to_rgb(image) * 255

	return rgb.astype("uint8")


def adjust_hsv(image, delta_h = 0, delta_s = 0, delta_v = 0):
	"""
	This function is to adjust the values in h,s, and v channels of an image

	Inputs:
		image: an non-normalized rgb image; 3-d numpy array
		delta_h: the amount to change the h channel; a float
		delta_s: the amount to change the s channel; a float
		delta_v: the amount to change the v channel; a float

	Output:
		image: the adjusted image; 3-d numpy array
	"""

	assert image.shape[-1] == 3
	assert 0 <= delta_h <= 1 and 0 <= delta_s <= 1 and 0 <= delta_v <= 1

	image = rgb_to_hsv(image / 255.0)

	image[:, :, 0] += delta_h
	image[:, :, 1] += delta_s
	image[:, :, 2] += delta_v

	image = hsv_to_rgb(image) * 255


	return image.astype("uint8")



def quantization(image, x_bins, y_bins, h_bins, s_bins, v_bins, delta = 1e-6):

	"""
	This function is quantize an RGB image converted to HSV space by (x, y, h, s, v)
	as described in the paper

	Inputs:
		image: a non-normalized rgb image; 3-d numpy array
		x_bins: the number of bins to which x coordinates are quantized; an int
		y_bins: the number of bins to which y coordinates are quantized; an int
		h_bins: the number of bins to which h values are quantized; an int
		s_bins: the number of bins to which s values are quantized; an int
		v_bins: the number of bins to which v values are quantized; an int
		delta: a small numerical value for computaional accuracy in the np.digitize function; float

	Output:
		output: the quantized values of (x, y, h, s, v); a 2-d numpy array

	"""

	assert image.ndim == 3

	h, w = image.shape[0], image.shape[1]

	image = rgb_to_hsv(image / 255)

	image_list = image.tolist()
	image_list = [j for i in image_list for j in i]

	## generate the list of coordinates
	## note: the coordinates are 0-index based here
	coordinates = [[i, j] for i in range(h) for j in range(w)]

	## merge hsv with coordinates
	image_list = list(zip(coordinates, image_list))
	image_list = [i[0] + i[1] for i in image_list]

	## normalize the coordinates
	image_list = [[i[0] / (h + delta - 1), i[1] / (w + delta - 1), i[2], i[3], i[4]] for i in image_list]


	## create the bins for quantization
	x_bin_array = np.linspace(0, 1, x_bins + 1)
	y_bin_array = np.linspace(0, 1, y_bins + 1)
	h_bin_array = np.linspace(0, 1, h_bins + 1)
	s_bin_array = np.linspace(0, 1, s_bins + 1)
	v_bin_array = np.linspace(0, 1, v_bins + 1)

	## quantization of the values
	quantized_list = [(x_bin_array[np.digitize(np.abs(i[0] - delta), x_bin_array) - 1],
					   y_bin_array[np.digitize(np.abs(i[1] - delta), y_bin_array) - 1],
					   h_bin_array[np.digitize(np.abs(i[2] - delta), h_bin_array) - 1],
					   s_bin_array[np.digitize(np.abs(i[3] - delta), s_bin_array) - 1],
					   v_bin_array[np.digitize(np.abs(i[4] - delta), v_bin_array) - 1]) for i in image_list]

	return quantized_list












def center_crop(img, size=256):
    height, width = img.shape[0], img.shape[1]

    if height < size or width < size:  # Upscale to size if one side is too small
        img = resize_to(img, resize=size)
        height, width = img.shape[0], img.shape[1]

    h_off = (height - size) // 2
    w_off = (width - size) // 2
    return img[h_off:h_off+size,w_off:w_off+size]







def resize_to(img, resize=512):
	'''Resize short side to target size and preserve aspect ratio'''
	height, width = img.shape[0], img.shape[1]
	if height < width:
	    ratio = height / resize
	    long_side = round(width / ratio)
	    resize_shape = (resize, long_side, 3)
	else:
	    ratio = width / resize
	    long_side = round(height / ratio)
	    resize_shape = (long_side, resize, 3)

	return scipy.misc.imresize(img, resize_shape, interp='bilinear')
























