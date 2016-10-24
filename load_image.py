import scipy.io
import scipy.misc
from skimage import data,img_as_ubyte
from PIL import Image
import numpy as np

MEAN_VALUE = np.array([0.48, 0.458, 0.407]).reshape((1,1,1,3))

def generate_noise_image(image,noise_ratio):
    noise_image = np.random.uniform(-0.5,0.5,
            (image.shape)).astype('float32')

    return noise_ratio*noise_image + (1-noise_ratio)*image

def load_image(path):
    image = scipy.misc.imread(path)
    image = image/255.0
    image = np.reshape(image,((1,)+image.shape)) - MEAN_VALUE
    
    return image

def save_image(image,path):
    img = image+MEAN_VALUE
    img = img[0]*255
    img = np.clip(img,0,255).astype('uint8')
    scipy.misc.imsave(path,img)


