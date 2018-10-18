#!/usr/bin/env python3
#
# @author abhay kumar

"""Preprocess images using Keras pre-trained models."""

import os

from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np
import pandas


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_pretrained_model(name):
    """
    name : pass the pretrained model name which you want to load. model name can be ('Xception','VGG16','VGG19','InceptionV3','MobileNet'). 
    	   default it returns resnet50 features. 
    returns the model
    """
    # include_top=False removes the fully connected layer at the end/top of the network
    # This allows us to get the feature vector as opposed to a classification
    if name == 'Xception':
        return applications.xception.Xception(weights='imagenet', include_top=False, pooling='avg')

    if name == 'VGG16':
        return applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')

    if name == 'VGG19':
        return applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')

    if name == 'InceptionV3':
        return applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    if name == 'MobileNet':
        return applications.mobilenet.MobileNet(weights='imagenet', include_top=False, pooling='avg')

    return applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')

def make_numpy_serialised(np_array):
    features_arr = np.char.mod('%f', np_array)
    return ','.join(features_arr)


def image_batch_generator(image_list,batch_size):
	for idx in range(0,len(image_list)-1,batch_size):
		yield image_list[idx:min(idx+batch_size,len(image_list))]



def extract_feature(model,img_path,serialised=False):
    """
    model : pass the loaded pretrained model .
    serialised : if enabled get the feature vector as string
    ret : returns the feature vector 
    """
    if os.path.isfile(img_path):
            print('is file: {}'.format(img_path))
            try:
                # load image setting the image size to 224 x 224
                img = image.load_img(img_path, target_size=(224, 224))
                # convert image to numpy array
                x = image.img_to_array(img)
                # the image is now in an array of shape (3, 224, 224)
                # but we need to expand it to (1, 3, 224, 224) as Keras is expecting a list of images
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                # extract the features
                features = model.predict(x)[0]
                if serialised:
                	return make_numpy_serialised(features)

                return features
            except Exception as ex:
                # skip all exceptions for now
                print(ex)
                pass

    return None

def main():
	a = ['a','b','c','d','e','f'] * 10

	print(next(image_batch_generator(a,3)))


if __name__=="__main__":
	import sys
	sys.exit(main())


