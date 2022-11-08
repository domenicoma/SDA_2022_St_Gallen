#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 21:22:54 2022

@author: ValerieLai
"""

"""
Fourier Transformation
"""
from PIL import Image
from numpy.fft import fft,ifft
import numpy as np
# Open the image by using Python Imaging Library(PIL)
image_before=Image.open('autumn.jpg')
# Decoding and encoding image to float number
"""
numpy. fromstring() function create a new one-dimensional array initialized from text data in a string
"""
image_int=np.fromstring(image_before.tobytes(), dtype=np.int8)
# Processing Fourier transform
fft_transformed=fft(image_int)
# Filter the lower frequency, i.e. employ a high pass
fft_transformed=np.where(np.absolute(fft_transformed) < 9e4,0,fft_transformed)
# Inverse Fourier transform
fft_transformed=ifft(fft_transformed)
# Keep the real part
fft_transformed=np.int8(np.real(fft_transformed))
# Output the image
image_output=Image.frombytes(image_before.mode, image_before.size, fft_transformed)
image_output.show()

"""
Filter 1: adjusting to 9e2
    adjusting 9e4 to 9e10 makes the entire image black
    adjusting 9e4 to 9e1 makes the image clearer
"""
image_before=Image.open('autumn.jpg')
image_int=np.fromstring(image_before.tobytes(), dtype=np.int8)
fft_transformed=fft(image_int)
fft_transformed=np.where(np.absolute(fft_transformed) < 9e2,0,fft_transformed)
fft_transformed=ifft(fft_transformed)
fft_transformed=np.int8(np.real(fft_transformed))
image_output=Image.frombytes(image_before.mode, image_before.size, fft_transformed)
image_output.show()

"""
Filter 2: adjusting to 9e6
"""
image_before=Image.open('autumn.jpg')
image_int=np.fromstring(image_before.tobytes(), dtype=np.int8)
fft_transformed=fft(image_int)
fft_transformed=np.where(np.absolute(fft_transformed) < 9e6,0,fft_transformed)
fft_transformed=ifft(fft_transformed)
fft_transformed=np.int8(np.real(fft_transformed))
image_output=Image.frombytes(image_before.mode, image_before.size, fft_transformed)
image_output.show()