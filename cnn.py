import keras
import tensorflow as tf
import numpy as np
from keras.applications import vgg16, vgg19, inception_v3, resnet50, mobilenet
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras import backend as K
from keras.models import Model

import os
from scipy import spatial
# from keras.applications.imagenet_utils import decode_predictions
# import matplotlib.pyplot as plt

# for Mac debug: cannot download keras models
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# FIRST METHOD

def load_model():
    #Load the VGG model
    print("---- LOADING MODELS FROM KERAS APPLICATIONS ----")
    vgg_model = vgg16.VGG16(weights='imagenet')
    # vgg_model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    # print(vgg_model.summary())

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
    return vgg_model

def load_single_img(path):
    """
    Return the numpy array of an image load from given path
    """
    # load an image in PIL format
    original = load_img(path, target_size=(224, 224))

    # convert the PIL image to a numpy array
    numpy_image_test = img_to_array(original)
    # plt.imshow(np.uint8(numpy_image))
    # plt.show()
    return numpy_image_test

# load all images from a directory
def load_img_from_folder(folder_path):
    print("---- LOAD ALL IMAGES FROM FOLDER ----")
    all_images = []
    images_info = []
    for img in os.listdir(folder_path):
        if img.split('.')[1] == 'jpg':
            name = img.split('.')[0]
            temp = {}
            temp['name'] = name
            images_info.append(temp)
            img = load_img(folder_path + '/' + img, target_size=(224, 224))
            img = img_to_array(img)
            # img = np.expand_dims(img, axis=0)
            all_images.append(img)
    return all_images,images_info

def convert_name_to_path(folder_path,images_info):
    images_path = []
    for img in images_info:
        path = folder_path + '/' + img['name'] + '.jpg'
        images_path.append(path)
    # print(images_path)
    return images_path

def compute_neighbor(vector_t,vector_c,k):
    """
    Return the index of k nearest neighbors of vector_t in order
    """
    # print("---- COMPUTE K NEAREST NEIGHBORHOOD ----")
    euclidean_distance = []
    cosine_distance = []
    for vec in vector_c:
        dis = np.linalg.norm(vector_t-vec)
        cdis = spatial.distance.cosine(vector_t, vec)
        euclidean_distance.append(dis)
        cosine_distance.append(cdis)

    dis_array = np.array(euclidean_distance)
    # idx_l2 = np.argpartition(dis_array, k)
    idx_l2 = np.argsort(dis_array)[:k]
    cdis_array = np.array(cosine_distance)
    # idx_cosine = np.argpartition(cdis_array,k)
    idx_cosine = np.argsort(cdis_array)[:k]
    return idx_l2, idx_cosine

def model_predict_method(tar,cand,k):
    print("---- EXTRACT FEATURE VECTORS FROM IMAGES ----")
    model = load_model()
    vector_c = []
    for img in cand:
        vector = model.predict(np.expand_dims(img, axis=0))
        vector_c.append(vector[0])
    vector_t = model.predict(np.expand_dims(tar, axis=0))
    return compute_neighbor(vector_t,vector_c,k)

# SECOND METHOD

def gram_matrix(feature,M,N):
    F = np.reshape(feature, (M, N))
    G = np.matmul(np.transpose(F), F)
    return G

def layer_difference(feature_a, feature_x):
    _, h, w, d = [i for i in feature_a.shape]
    M = h * w
    N = d
    A = gram_matrix(feature_a, M, N)
    G = gram_matrix(feature_x, M, N)

    total = 0
    for i in range(N):
        for j in range(N):
            total += (A[i][j]-G[i][j])**2
    return total/(4 * (N**2) * (M**2))

def style_feature_method(tar,cand,k):
    print("---- EXTRACT STYLE FEATURES FROM IMAGES ----")
    layers_style = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    layers_name = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

    layers_style_weights = [0.2,0.2,0.2,0.2,0.2]
    base_model = vgg19.VGG19(weights='imagenet')

    def preprocess(img_path):
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = vgg19.preprocess_input(x)
        return x

    ptar = preprocess(tar)

    for i in range(len(layers_name)):
        model = Model(inputs=base_model.input, outputs=base_model.get_layer(layers_name[i]).output)
        feature_a = model.predict(ptar)
        all_loss = [0]*len(cand)

        print("Extracting style features from "+ layers_name[i])

        for j in range(len(cand)):
            pcand = preprocess(cand[j])
            feature_x = model.predict(pcand)
            loss = layer_difference(feature_a, feature_x)
            all_loss[j] += loss*layers_style_weights[i]

    loss_array = np.array(all_loss)
    idx_loss = np.argsort(loss_array)[:k]
    return idx_loss

# print(test_prediction.shape) #(1,1000) one image becomes a 1000 length feature vector
# print(all_vectors[0])

# compute_neighbor()

# # Convert the image / images into batch format
# # expand_dims will add an extra dimension to the data at a particular axis
# # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
# # Thus we add the extra dimension to the axis 0.
# image_batch = np.expand_dims(numpy_image, axis=0)
# print('image batch size', image_batch.shape)
# plt.imshow(np.uint8(image_batch[0]))
