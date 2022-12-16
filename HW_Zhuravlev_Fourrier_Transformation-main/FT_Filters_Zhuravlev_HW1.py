#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:35:56 2022

@author: Vasily Zhuravlev

Template code: original code taken from Junjie Hu and Isabell Fetzer, 20220924 WKH 

Homework 1: SDA 2022 HSG
"""



"""
Fourier Transformation
"""
#Import the neceassary library

from PIL import Image
from numpy.fft import fft,ifft
import numpy as np

#First image transformation


# Open the image by using Python Imaging Library(PIL)
image_before=Image.open('dentsdumidi_avant.jpg')
# Decoding and encoding image to float number
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
image_output.save("dentsdumidi_apres.jpg")


# Second image transformation using a slightly modified filter



# Open the image by using Python Imaging Library(PIL)
image_before=Image.open('zlatni_rat_do.jpg')
# Decoding and encoding image to float number
image_int=np.fromstring(image_before.tobytes(), dtype=np.int8)
# Processing Fourier transform
fft_transformed=fft(image_int)
# Filter the lower frequency, i.e. employ a high pass
fft_transformed=np.where(np.absolute(fft_transformed) < 9e5,0,fft_transformed)
# Inverse Fourier transform
fft_transformed=ifft(fft_transformed)
# Keep the real part
fft_transformed=np.int8(np.real(fft_transformed))
# Output the image
image_output=Image.frombytes(image_before.mode, image_before.size, fft_transformed)
image_output.show()
image_output.save("zlatni_rat_posle_9e5.jpg")



# Same image of the beach transformed using a slightly modified filter, lower frequency


# Open the image by using Python Imaging Library(PIL)
image_before=Image.open('zlatni_rat_do.jpg')
# Decoding and encoding image to float number
image_int=np.fromstring(image_before.tobytes(), dtype=np.int8)
# Processing Fourier transform
fft_transformed=fft(image_int)
# Filter the lower frequency, i.e. employ a high pass
fft_transformed=np.where(np.absolute(fft_transformed) < 9e6,0,fft_transformed)
# Inverse Fourier transform
fft_transformed=ifft(fft_transformed)
# Keep the real part
fft_transformed=np.int8(np.real(fft_transformed))
# Output the image
image_output=Image.frombytes(image_before.mode, image_before.size, fft_transformed)
image_output.show()
image_output.save("zlatni_rat_posle_9e6.jpg")
