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
# from tqdm import tqdm
import re
# import cProfile
# import pstats
import os
import subprocess

REAL_RANGE_MIN = -1.0
REAL_RANGE_MAX = 1.0
IMAG_RANGE_MIN = -1.3
IMAG_RANGE_MAX = 1.3

IMAGE_WIDTH = 4000
IMAGE_HEIGHT = (IMAG_RANGE_MAX - IMAG_RANGE_MIN) * (IMAGE_WIDTH) / (REAL_RANGE_MAX - REAL_RANGE_MIN)
DIVERGENCE_LIMIT = np.inf
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

class JuliaSetGenerator:
    def __init__(self, file_name: str, expression: str, real_range_min: float = -2.31, real_range_max: float = 2.31, imag_range_min: float = -1.3, imag_range_max: float = 1.3, image_width=1920, image_height=1080, label_image=False, iteration_count=10, cmap='gnuplot'):
        self.filename = file_name
        self.expression = expression
        self.real_range_min = real_range_min
        self.real_range_max = real_range_max
        self.imag_range_min = imag_range_min
        self.imag_range_max = imag_range_max
        self.image_width = image_width
        self.image_height = image_height
        self.label_image = label_image
        self.iteration_count = iteration_count
        self.cmap = cmap
        self.vect_divergence_tracker = np.vectorize(divergence_tracker)
        
        try:
            subprocess.check_output('nvidia-smi')
            subprocess.check_output('nvcc --version')
            self.has_gpu = True
        except Exception:
            self.has_gpu = False
    
    def save_julia_set_image(self):
        if(not is_safe(self.expression)):
            raise ValueError('invalid expression')
        sympy_func = sympify(self.expression)
        
        math_lambda_func = lambdify((x,), sympy_func, 'math')
        if(self.has_gpu):
            cuda_func = cuda.jit('void(complex64)', device=True)(math_lambda_func)
            cuda_divergence_tracker = cuda.jit('void(complex64, int32)', device=True)(divergence_tracker)
            @cuda.jit('void(complex64[:,:], int32[:,:], int32)')
            def cuda_calculate_julia_set(x, divergence, iter):
                xstart, ystart = cuda.grid(2)
                xstride, ystride = cuda.gridsize(2)
                for k in range(iter):
                    for i in range(xstart, x.shape[0], xstride):
                        for j in range(ystart, x.shape[1], ystride):
                            if(x[i, j].real**2 + x[i, j].imag**2 < DIVERGENCE_LIMIT):
                                x[i, j] = cuda_func(x[i, j])
                            divergence[i, j] = cuda_divergence_tracker(x[i, j], divergence[i, j])
            img = self.plotting_helper(cuda_calculate_julia_set)
        else:       
            def calculate_julia_set(x, divergence):
                for iteration in range(self.iteration_count):
                    if x.real**2 + x.imag**2 < DIVERGENCE_LIMIT:
                        divergence = divergence + 1
                        x = math_lambda_func(x)
                return divergence
            vect_calculate_julia_set = np.vectorize(calculate_julia_set)        
            
            img = self.plotting_helper(vect_calculate_julia_set)
        return img
    
    def plotting_helper(self, expression):
        norm = matplotlib.colors.Normalize(vmin=0, vmax=self.iteration_count)
        normalized = norm(self.plot_julia_set_image(expression))

        color_map = matplotlib.cm.get_cmap(self.cmap)
        img = Image.fromarray(np.uint8(color_map(normalized) * 255))
        if(self.label_image):
            I1 = ImageDraw.Draw(img)
            my_font = ImageFont.truetype('arial', 20)
            I1.text((28, 36), self.expression, fill=(255, 0, 0), font=my_font)
        return img
    
    def plot_julia_set_image(self, expression):
        """
        takes complex expression, iterates the expression with a set of initial complex points,
        with each point corresponding to a pixel in the resulting image,
        and returns the times it was iterated before the value diverged.

        Args:
            expression (callable): julia set function
        """
        real_set = np.linspace(self.real_range_min, self.real_range_max, int(self.image_width)).reshape((1, int(self.image_width)))
        imag_set = np.linspace(self.imag_range_max, self.imag_range_min, int(self.image_height)).reshape((int(self.image_height), 1))
        complex_set = np.array(real_set + 1j * imag_set, dtype=np.complex64)

        divergence = self.iterations_till_divergence_image(expression, complex_set)
        return divergence
    
    def iterations_till_divergence_image(self, expression, initial_values):
        """ iterates over the initial values with a function and returns
        a list of when the modulus of each complex value goes above a value

        Args:
            expression (ufunc): function of x to be iterated
            initial_values (np.complex64[][]): array of initial values
            a (float): optional variable

        Returns:
            list: list of when each value diverges
        """
        divergence_h = np.zeros(initial_values.shape, dtype=np.int32)

        if(self.has_gpu):
            blockdim = (16, 16)
            griddimx = math.ceil(self.image_width / blockdim[0])
            griddimy = math.ceil(self.image_height / blockdim[1])
            griddim = (griddimx, griddimy)

            iterating_values_d = cuda.to_device(initial_values)
            divergence_d = cuda.to_device(divergence_h)

            expression[griddim, blockdim](iterating_values_d, divergence_d, self.iteration_count)
            divergence_h = divergence_d.copy_to_host()
        else:
            divergence_h = expression(initial_values, divergence_h)
        return divergence_h

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

def save_img_to_file(image, filename):
    # background = Image.new("RGB", image.size, (255, 255, 255))
    # background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
    image.save(fp=filename, format='PNG')


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
    pass

if __name__ == '__main__':
    main()
