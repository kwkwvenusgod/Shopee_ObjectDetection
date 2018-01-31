from __future__ import division
import random
import cv2
import numpy as np
import sys
import pickle
import load_data
import simplejson

from pathlib import Path
from keras import backend as K
from keras_frcnn import losses as losses
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras_frcnn import data_generators

sys.setrecursionlimit(40000)
with open('train_dataset_config.json', 'rb') as dataset_config_file:
    dataset_config = simplejson.load(dataset_config_file)

all_imgs, classes_count, class_mapping = load_data.load(dataset_config, size_lim=10000)
with open('train_dataset_config.json', 'rb') as dataset_config_file:
    dataset_config = simplejson.load(dataset_config_file)

with open('model_output/config.pickle', 'rb') as f_in:
    frcnn_config = pickle.load(f_in)


if frcnn_config.network == 'vgg':
    frcnn_config.network = 'vgg'
    from keras_frcnn import vgg as nn
elif frcnn_config.network == 'resnet50':
    from keras_frcnn import resnet as nn
    frcnn_config.network = 'resnet50'
else:
    print('Not a valid model')
    raise ValueError

class_mapping = frcnn_config.class_mapping

inv_map = {v: k for k, v in class_mapping.items()}
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']
print('Num val samples {}'.format(len(val_imgs)))

data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, frcnn_config, nn.get_img_output_length,
                                             K.image_dim_ordering(), mode='val')

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(frcnn_config.anchor_box_scales) * len(frcnn_config.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, frcnn_config.num_rois, nb_classes=len(class_mapping),
                           trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

model_path = 'model_output/'+frcnn_config.model_path
pretrain_model_path = 'pretrain/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

try:
    if Path(model_path).exists() is True:
        model_all.load_weights(filepath=model_path, by_name=True)
        model_rpn.load_weights(filepath=model_path, by_name=True)
        model_classifier.load_weights(filepath=model_path, by_name=True)
        print("Succesfully load model parameters")
    elif Path(pretrain_model_path).exists() is True:
        model_rpn.load_weights(filepath=pretrain_model_path, by_name=True)
        model_classifier.load_weights(filepath=pretrain_model_path, by_name=True)
        print("Succesfully load pretrained standard keras model parameters")
except:
    print("No pre trained model")

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier,
                         loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count) - 1)],
                         metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')


inx = 0
losses = np.zeros((len(val_imgs), 5))
for val_image in val_imgs:
    X, Y, img_data = next(data_gen_val)
    res_rpn = model_rpn.evaluate(X, Y, batch_size=1)

    pred_rpn = model_rpn.predict(X)

    R = roi_helpers.rpn_to_roi(pred_rpn[0], pred_rpn[1], frcnn_config, K.image_dim_ordering(), use_regr=True,
                               overlap_thresh=0.7,
                               max_boxes=300)
    # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
    X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, frcnn_config, class_mapping)

    if X2 is None:
        continue


    pred_classifier = model_classifier.predict_on_batch([X, X2])
    res_classifier = model_classifier.evaluate([X, X2],[Y1, Y2[:, :, 64:]])

    losses[inx, 0] = res_rpn[1]
    losses[inx, 1] = res_rpn[2]

    losses[inx, 2] = res_classifier[1]
    losses[inx, 3] = res_classifier[2]
    losses[inx, 4] = res_classifier[3]

    print('Loss RPN classifier: {}'.format(res_rpn[1]))
    print('Loss RPN regression: {}'.format(res_rpn[2]))
    print('Loss Detector classifier: {}'.format(res_classifier[1]))
    print('Loss Detector regression: {}'.format(res_classifier[2]))
    inx += 1


print('Classifier accuracy for bounding boxes from RPN: {}'.format(np.mean(losses[:,4])))
print('Loss RPN classifier: {}'.format(np.mean(losses[:,0])))
print('Loss RPN regression: {}'.format(np.mean(losses[:,1])))
print('Loss Detector classifier: {}'.format(np.mean(losses[:,2])))
print('Loss Detector regression: {}'.format(np.mean(losses[:,3])))

