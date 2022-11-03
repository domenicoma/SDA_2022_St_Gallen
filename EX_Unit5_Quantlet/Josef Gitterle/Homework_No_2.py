"""
DEDA Unit 5 - Homework No.2
Basic Statistics and Visualisation in Python
Authors: Junjie Hu and Isabell Fetzer, 20220924 WKH 
"""


"""
Interactive Graphs
"""
"""
Plotly enables Python users to create beautiful interactive visualisations.
"""
# pip install yfinance
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import numpy as np


"""
Fourier Transformation
"""
from PIL import Image
from numpy.fft import fft,ifft
import numpy as np
# Open the image by using Python Imaging Library(PIL)
image_before=Image.open('HWSY5545.jpg')
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
