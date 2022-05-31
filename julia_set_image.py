"""Julia set image
    Class: CPSC 455
    By: Nathan Flack
    Version: 1.0
"""
from __future__ import division

import logging
import math
import time

import matplotlib
import numpy as np
import psutil
from numba import cuda
from PIL import Image, ImageDraw, ImageFont
from sympy import *
from tqdm import tqdm
import re
# import cProfile
# import pstats
import os

REAL_RANGE_MIN = -1.0
REAL_RANGE_MAX = 1.0
IMAG_RANGE_MIN = -1.3
IMAG_RANGE_MAX = 1.3

IMAGE_WIDTH = 4000
IMAGE_HEIGHT = (IMAG_RANGE_MAX - IMAG_RANGE_MIN) * (IMAGE_WIDTH) / (REAL_RANGE_MAX - REAL_RANGE_MIN)
DIVERGENCE_LIMIT = 4000

FRAMES = 32
DURATION = 80
ITERATIONS = 90

FILENAME = f'pictures/stills1/still3.png'

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -43s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')

# norm = matplotlib.colors.Normalize(vmin=0, vmax=200)

x, y, z, t, a = symbols('x y z t a')
# examples:
# 1 - x**2 + x**2 / (2 + 4 * x) + 0.7885 * np.e**(a * 1j)
# 1 - x + x**2 + 0.7885 * np.e**(a * 1j)
# x**4 + x**3/(x-1) + x**2/(x**3 + 4 *x**2 + 5) - 0.5885 * np.e**(a * 1j)
# x**4 + x**3/(x-1) + x**2/(x**3 + 4 *x**2 + 5) + 0.755534*math.cos(a) + 0.737292*1j*math.cos(a) - 2*0.737292*1j
# 2**x + 0.2885 * np.e**(a * 1j)
# x**2 + 0.355534*math.cos(2*a)-0.337292*1j*math.cos(a)
# x**4 + x**3 / (x - 1) + x**2 / (x**3 + 4 * x**2 + 5) + 0.377767 * math.sin(a) + 0.368646 * 1j * math.sin(a) - 0.368646 * 1j + 0.377767
# x**2 + a*.01 - a*.3*1j
# compatible color maps can be found at https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html

@cuda.jit('void(complex64, float32)', device=True)
def iterating_function(x: np.complex64, a: np.float32):
    """
    cuda device function
    takes an 'x' value and 'a' value and returns the calculated result

    Args:
        x (complex64): variable1
        a (float32): variable2

    Returns:
        complex64: result
    """
    return x**4 + x**3 / (x - 1) + x**2 / (x**3 + 4 * x**2 + 5) + 0.377767 * math.sin(a) + 0.368646 * 1j * math.sin(a) - 0.368646 * 1j + 0.377767


@cuda.jit('void(complex64, int32)', device=True)
def divergence_tracker(x: np.complex64, divergence: np.int32):
    """
    cuda device function
    checks whether the x value is diverging and increase divergence by 1 if the x value is not diverging

    Args:
        x (complex64): value to check
        divergence (int32): current value in divergence

    Returns:
        int32: new divergence value
    """
    if x.real**2 + x.imag**2 < DIVERGENCE_LIMIT:
        return divergence + 1
    return divergence


@cuda.jit('void(complex64[:,:], int32[:,:], float32, int32)')
def helper_func5_image(x, divergence, a, iter):
    """
    full cuda kernel. compiled by numba cuda jit from python to a gpu compatible function

    Args:
        x (complex64[][]): array of complex values to iterate
        divergence (int32[][]): divergence tracking array
        a (float32): optional variable
        iter (int32): amount of times to iterate
    """
    xstart, ystart = cuda.grid(2)
    xstride, ystride = cuda.gridsize(2)
    for k in range(iter):
        for i in range(xstart, x.shape[0], xstride):
            for j in range(ystart, x.shape[1], ystride):
                if(x[i, j].real**2 + x[i, j].imag**2 < DIVERGENCE_LIMIT):
                    x[i, j] = iterating_function(x[i, j], a)
                divergence[i, j] = divergence_tracker(x[i, j], divergence[i, j])


def plot_julia_set_image(expression: callable, a, iteration_count: int = 100, real_range_min: float = -1.0, real_range_max: float = 1.0,
                         imag_range_min: float = -1.0, imag_range_max: float = 1.0, image_width=1920, image_height=1080):
    """
    takes complex expression, iterates the expression with a set of initial complex points,
    with each point corresponding to a pixel in the resulting image,
    and returns the times it was iterated before the value diverged.

    Args:
        expression (callable): julia set cuda function
        iteration_count (int): amount of times to iterate
        seed_count (int): amount of points to iterate and plot
        real_range_min (float): min range of real part of complex points
        real_range_max (float): max range of real part of complex points
        imag_range_min (float): min range of imag part of complex points
        imag_range_max (float): max range of imag part of complex points
        image_width (int): array width of point set
        image_height (int): array height of point set
    """
    # start_time = time.perf_counter()

    real_set = np.linspace(real_range_min, real_range_max, int(image_width)).reshape((1, int(image_width)))
    imag_set = np.linspace(imag_range_max, imag_range_min, int(image_height)).reshape((int(image_height), 1))
    complex_set = np.array(real_set + 1j * imag_set, dtype=np.complex64)

    divergence = iterations_till_divergence_image(expression, complex_set, a, iteration_count)

    # LOGGER.info(f'time taken plot_julia_set: {time.perf_counter() - start_time}')
    return divergence


def iterations_till_divergence_image(expression: callable, initial_values: np.ndarray, a: float, iteration_count=1000) -> list:
    """ iterates over the initial values with a function and returns
        a list of when the modulus of each complex value goes above a value

    Args:
        expression (ufunc): function of x to be iterated
        initial_values (np.complex64[][]): array of initial values
        a (float): optional variable
        max_iterations (int): amount of iterations

    Returns:
        list: list of when each value diverges
    """
    # start_time = time.perf_counter()
    divergence_h = np.zeros(initial_values.shape, dtype=np.int32)

    blockdim = (16, 16)
    griddimx = math.ceil(IMAGE_WIDTH / blockdim[0])
    griddimy = math.ceil(IMAGE_HEIGHT / blockdim[1])
    griddim = (griddimx, griddimy)

    iterating_values_d = cuda.to_device(initial_values)
    divergence_d = cuda.to_device(divergence_h)

    expression[griddim, blockdim](iterating_values_d, divergence_d, a, iteration_count)
    divergence_h = divergence_d.copy_to_host()

    # LOGGER.info(f'time taken iterations_till_divergence_image: {time.perf_counter() - start_time}')
    return divergence_h


def save_julia_set_over_time_image(file_name: str, expression: str, real_range_min: float = -2.31, real_range_max: float = 2.31, imag_range_min: float = -1.3, imag_range_max: float = 1.3, image_width=1920, image_height=1080, label_image=False, frames=10):
    """ generates many julia set images with the a variable changing gradually
        between each image to create a slightly different image.
        These images are then combined into a gif and saved.

    Args:
        file_name (str): name of file location to save gif at
        expression (str): string representation of expression. currently only used for image text so it wont change the julia set for now
        real_range_min (float, optional): minimum real. Defaults to -2.31.
        real_range_max (float, optional): maxiimum real. Defaults to 2.31.
        imag_range_min (float, optional): minimum imaginary. Defaults to -1.3.
        imag_range_max (float, optional): maximum imaginary. Defaults to 1.3.
        image_width (int, optional): pixel width of the image. Defaults to 1920.
        image_height (int, optional): pixel height of the image. Defaults to 1080.
        label_image (bool, optional): whether to place expression into the image. Defaults to False.
        frames (int, optional): how many frames to put in gif. Defaults to 10.

    Returns:
        _type_: _description_
    """
    # start_time = time.perf_counter()

    imgs = []
    for a in np.linspace(0, 2 * np.pi, frames):
        if psutil.virtual_memory().available < 200000000:
            break
        img = plotting_helper(iterating_function, a, 200, real_range_min, real_range_max, imag_range_min, imag_range_max, image_width, image_height, label_image, expression)
        imgs.append(img)

    img = imgs[0]  # extract first image from iterator
    img.save(fp=file_name, format='GIF', append_images=imgs[1:],
             save_all=True, duration=DURATION, loop=0)
    return imgs


def save_julia_set_image(file_name: str, expression: str, real_range_min: float = -2.31, real_range_max: float = 2.31, imag_range_min: float = -1.3, imag_range_max: float = 1.3, image_width=1920, image_height=1080, label_image=False, a=0, iteration_count=200, cmap='gnuplot'):
    if(not is_safe(expression)):
        raise ValueError('invalid expression')
    sympy_func = sympify(expression)
    lambda_func = lambdify((x, y), sympy_func, 'math')
    cuda_func = cuda.jit('void(complex64, float32)', device=True)(lambda_func)

    @cuda.jit('void(complex64[:,:], int32[:,:], float32, int32)')
    def calculate_julia_set(x, divergence, a, iter):
        xstart, ystart = cuda.grid(2)
        xstride, ystride = cuda.gridsize(2)
        for k in range(iter):
            for i in range(xstart, x.shape[0], xstride):
                for j in range(ystart, x.shape[1], ystride):
                    if(x[i, j].real**2 + x[i, j].imag**2 < DIVERGENCE_LIMIT):
                        x[i, j] = cuda_func(x[i, j], a)
                    divergence[i, j] = divergence_tracker(x[i, j], divergence[i, j])

    img = plotting_helper(calculate_julia_set, a, iteration_count, real_range_min, real_range_max, imag_range_min, imag_range_max, image_width, image_height, label_image, expression, cmap)
    return img

def save_img_to_file(image, filename):
    # background = Image.new("RGB", image.size, (255, 255, 255))
    # background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
    image.save(fp=filename, format='PNG')

def plotting_helper(expression: callable, a, iteration_count: int = 100, real_range_min: float = -2.31, real_range_max: float = 2.31, imag_range_min: float = -1.3, imag_range_max: float = 1.3, image_width=1920, image_height=1080, label_image=False, function_string='', cmap='gnuplot'):
    norm = matplotlib.colors.Normalize(vmin=0, vmax=iteration_count+1)
    normalized = norm(plot_julia_set_image(expression, a, iteration_count, real_range_min, real_range_max, imag_range_min, imag_range_max, image_width, image_height))

    color_map = matplotlib.cm.get_cmap(cmap)
    img = Image.fromarray(np.uint8(color_map(normalized) * 255))
    if(label_image):
        I1 = ImageDraw.Draw(img)
        my_font = ImageFont.truetype('arial', 20)
        I1.text((28, 36), function_string, fill=(255, 0, 0), font=my_font)
    return img

def _is_invalid(c):
    return c.isalpha() or c == '_'

def is_safe(inp_string):
    """ Blacklist attribute access, simply by checking for any period that is
    not surrounded by numbers. Returns True for '3.4', but not for 'a.b' """
    # components = inp_string.split(".")
    after = re.findall(r'(?<=\.)[^\s]', inp_string) # gets the non whitespace on right of period
    before = re.findall(r'[^\s](?=\.)', inp_string) # gets the non whitespace on left of period
    components = list(before) + list(after)
    if len(components) == 1:
        return True
    for c in components:
        if _is_invalid(c):
            return False
    return True

def main():
    # logging.basicConfig(level=logging.DEBUG,
    #                     format=LOG_FORMAT, filename='log_file_test.log')
    # logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    # logging.getLogger('numba.core').setLevel(logging.WARNING)
    # logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)

    
    # strfunc = 'x**4 + x**3 / (x - 1) + x**2 / (x**3 + 4 * x**2 + 5) + 0.377767 * sin(.5) + 0.368646 * 1j * sin(2.6) - 0.368646 * 1j + 0.377767'
    # cProfile.run(f'save_julia_set_image(FILENAME, \'{strfunc}\', REAL_RANGE_MIN, REAL_RANGE_MAX, IMAG_RANGE_MIN, IMAG_RANGE_MAX, IMAGE_WIDTH, IMAGE_HEIGHT, True)', 'log_file_test.log') 
    # save_julia_set_image(f'pictures/sequence/julia{len(os.listdir(os.path.join(os.getcwd(), "pictures/sequence")))+1}.png', strfunc, REAL_RANGE_MIN, REAL_RANGE_MAX, IMAG_RANGE_MIN, IMAG_RANGE_MAX, IMAGE_WIDTH, IMAGE_HEIGHT, True, iteration_count= ITERATIONS,cmap='hsv')
    # p = pstats.Stats('log_file_test.log')
    # p.sort_stats('cumulative').print_stats(20)

    ...

if __name__ == '__main__':
    # multiprocessing.freeze_support()
    main()
