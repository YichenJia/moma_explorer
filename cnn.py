import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.preprocessing.image import load_img, img_to_array
import os
from scipy import spatial
# from keras.applications.imagenet_utils import decode_predictions
# import matplotlib.pyplot as plt

# for Mac debug: cannot download keras models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#Load the VGG model
print("---- LOADING MODELS FROM KERAS APPLICATIONS ----")
vgg_model = vgg16.VGG16(weights='imagenet')
# vgg_model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
print(vgg_model.summary())

#Load the Inception_V3 model
# inception_model = inception_v3.InceptionV3(weights='imagenet')
# print(inception_model.summary())
#
# #Load the ResNet50 model
# resnet_model = resnet50.ResNet50(weights='imagenet')
# print(resnet_model.summary())
#
# # #Load the MobileNet model
# mobilenet_model = mobilenet.MobileNet(weights='imagenet')
# print(mobilenet_model.summary())

# load an image in PIL format
# filename = 'MOMA/2.jpg'
# original = load_img(filename, target_size=(224, 224))
# convert the PIL image to a numpy array
# numpy_image = img_to_array(original)
# plt.imshow(np.uint8(numpy_image))
# plt.show()
# print('numpy array size',numpy_image.shape)


# load all images from a directory
print("---- LOAD ALL IMAGES FROM FOLDER ----")
all_images = []
images_info = []
folder_path = "test_MOMA"
count = 0
for img in os.listdir(folder_path):
    name = img.split('.')[0]
    temp = {}
    temp['name'] = name
    images_info.append(temp)
    img = load_img(folder_path + '/' + img, target_size=(224, 224))
    img = img_to_array(img)
    # img = np.expand_dims(img, axis=0)
    all_images.append(img)

# all_images = np.asarray(all_images)
# print(all_images.shape)
print(images_info)

print("---- EXTRACT FEATURE VECTORS FROM IMAGES ----")
all_vectors = []
for img in all_images:
    vector = vgg_model.predict(img.reshape([1,224,224,3]))
    all_vectors.append(vector[0])


print("ONE SAMPLE PREDICTION: ")
# print(test_prediction.shape) #(1,1000) one image becomes a 1000 length feature vector
# print(all_vectors[0])

print("---- COMPUTE K NEAREST NEIGHBORHOOD ----")
euclidean_distance = []
cosine_distance = []
target_vector = all_vectors[0]
for vec in all_vectors:
    dis = np.linalg.norm(target_vector-vec)
    cdis = spatial.distance.cosine(target_vector, vec)
    euclidean_distance.append(dis)
    cosine_distance.append(cdis)
print("ALL EUCLIDEAN DISTANCES: ")
print(euclidean_distance)
print("ALL COSINE DISTANCES: ")
print(cosine_distance)


# # Convert the image / images into batch format
# # expand_dims will add an extra dimension to the data at a particular axis
# # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
# # Thus we add the extra dimension to the axis 0.
# image_batch = np.expand_dims(numpy_image, axis=0)
# print('image batch size', image_batch.shape)
# plt.imshow(np.uint8(image_batch[0]))

# all_weights = []
# for layer in vgg_model.layers:
#    w = layer.get_weights()
#    all_weights.append(w)
# print("All Weights:")
# print(all_weights)
