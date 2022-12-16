[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **filter** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: 'filter'

Published in: 'SDA_2022_St_Gallen'

Description: 'Creating a filter using Fourier Transformation for a picture of a snake or data center'

Keywords: 'Fourier Transformation, Filter, Picture, Fourier, Snake, Data Center'

Author: 'Raul Bag'

Submitted: '03 November 2022'

Input: 'snake.png'

Output: 'snake.png'


```

![Picture1](data_center.jpg)

![Picture2](snake.png)

### PYTHON Code
```python

"""
Fourier Transformation
"""
from PIL import Image
from numpy.fft import fft,ifft
import numpy as np

# Open the image by using Python Imaging Library(PIL)
#image_before=Image.open('data_center.jpg')
image_before=Image.open('snake.png')
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

```

automatically created on 2022-11-06