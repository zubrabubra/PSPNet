# -*- coding: utf-8 -*-
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from generator_mask import data_gen_small

import os
import numpy as np
import pandas as pd
import argparse
import json
import cv2
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from PSPNet import PSPNet50

train_path='../../data/stage1_train/'
test_path='../../data/stage1_test/'

# for generator
batch_size = 2

def make_df(train_path='../../data/stage1_train/', test_path='../../data/stage1_test/', img_size=256):
    train_ids = next(os.walk(train_path))[1]
    test_ids = next(os.walk(test_path))[1]
    X_train = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)
    print('=== call train ===')
    for i, id_ in tqdm(enumerate(train_ids)):
        path = train_path + id_
        img = cv2.imread(path + '/images/' + id_ + '.png')[:,:,::-1]
        img = cv2.resize(img, (img_size, img_size))
        X_train[i] = img_to_array(img)/256
        mask = np.zeros((img_size, img_size, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
            mask_ = cv2.resize(mask_, (img_size, img_size))
            mask_ = mask_[:, :, np.newaxis]
            mask = np.maximum(mask, mask_)
        Y_train[i] = img_to_array(mask)
    X_test = np.zeros((len(test_ids), img_size, img_size, 3), dtype=np.uint8)
    sizes_test = []
    print('=== call test ===')
    for i, id_ in tqdm(enumerate(test_ids)):
        path = test_path + id_
        img = cv2.imread(path + '/images/' + id_ + '.png')
        sizes_test.append([img.shape[0], img.shape[1]])
        img = cv2.resize(img, (img_size, img_size))
        X_test[i] = img_to_array(img)/256

    return X_train, Y_train, X_test, sizes_test


from keras.preprocessing.image import ImageDataGenerator


def generator(xtr, xval, ytr, yval, batch_size):
    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(xtr, seed=7)
    mask_datagen.fit(ytr, seed=7)
    image_generator = image_datagen.flow(xtr, batch_size=batch_size, seed=7)
    mask_generator = mask_datagen.flow(ytr, batch_size=batch_size, seed=7)
    train_generator = zip(image_generator, mask_generator)

    val_gen_args = dict()
    image_datagen_val = ImageDataGenerator(**val_gen_args)
    mask_datagen_val = ImageDataGenerator(**val_gen_args)
    image_datagen_val.fit(xval, seed=7)
    mask_datagen_val.fit(yval, seed=7)
    image_generator_val = image_datagen_val.flow(xval, batch_size=batch_size, seed=7)
    mask_generator_val = mask_datagen_val.flow(yval, batch_size=batch_size, seed=7)
    val_generator = zip(image_generator_val, mask_generator_val)

    return train_generator, val_generator

if __name__ == "__main__":
    # command line argments
    parser = argparse.ArgumentParser(description="SegUNet LIP dataset")
    parser.add_argument("--train_list",
            default="../LIP/TrainVal_images/train_id.txt",
            help="train list path")
    parser.add_argument("--trainimg_dir",
            default="../LIP/TrainVal_images/TrainVal_images/train_images/",
            help="train image dir path")
    parser.add_argument("--trainmsk_dir",
            default="../LIP/TrainVal_parsing_annotations/TrainVal_parsing_annotations/train_segmentations/",
            help="train mask dir path")
    parser.add_argument("--val_list",
            default="../LIP/TrainVal_images/val_id.txt",
            help="val list path")
    parser.add_argument("--valimg_dir",
            default="../LIP/TrainVal_images/TrainVal_images/val_images/",
            help="val image dir path")
    parser.add_argument("--valmsk_dir",
            default="../LIP/TrainVal_parsing_annotations/TrainVal_parsing_annotations/val_segmentations/",
            help="val mask dir path")
    parser.add_argument("--batch_size",
            default=10,
            type=int,
            help="batch size")
    parser.add_argument("--n_epochs",
            default=30,
            type=int,
            help="number of epoch")
    parser.add_argument("--epoch_steps",
            default=3000,
            type=int,
            help="number of epoch step")
    parser.add_argument("--val_steps",
            default=500,
            type=int,
            help="number of valdation step")
    parser.add_argument("--n_labels",
            default=1,
            type=int,
            help="Number of label")
    parser.add_argument("--input_shape",
            default=(256, 256, 3),
            help="Input images shape")
    parser.add_argument("--output_stride",
            default=16,
            type=int,
            help="output_stride")
    parser.add_argument("--output_mode",
            default="sigmoid",
            type=str,
            help="output mode")
    parser.add_argument("--upsample_type",
            default="deconv",
            type=str,
            help="upsampling type")
    parser.add_argument("--loss",
            default="binary_crossentropy",
            type=str,
            help="loss function")
    parser.add_argument("--optimizer",
            default="adadelta",
            type=str,
            help="oprimizer")
    parser.add_argument("--gpu_num",
            default="0",
            type=str,
            help="num of gpu")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    # set the necessary list
    #train_list = pd.read_csv(args.train_list,header=None)
    #val_list = pd.read_csv(args.val_list,header=None)

    # set the necessary directories
    trainimg_dir = args.trainimg_dir
    trainmsk_dir = args.trainmsk_dir
    valimg_dir = args.valimg_dir
    valmsk_dir = args.valmsk_dir

    # get old session
    old_session = KTF.get_session()

    with tf.Graph().as_default():
        session = tf.Session('')
        KTF.set_session(session)
        KTF.set_learning_phase(1)

        # set callbacks
        fpath = '../../data/PSPNet50_mask{epoch:02d}.hdf5'
        cp_cb = ModelCheckpoint(filepath = fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=5)
        es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
        tb_cb = TensorBoard(log_dir="../../data/pretrained_mask", write_images=True)

        X_train, Y_train, X_test, sizes_test = make_df()
        xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1, random_state=7)
        train_gen, val_gen = generator(xtr, xval, ytr, yval, batch_size)

        # set model
        pspnet = PSPNet50(input_shape=args.input_shape,
                n_labels=args.n_labels,
                output_stride=8,
                levels=[3,2,1],
                output_mode=args.output_mode,
                upsample_type=args.upsample_type)
        print(pspnet.summary())

        # compile model
        pspnet.compile(loss=args.loss,
                optimizer=args.optimizer,
                metrics=["accuracy"])

        # fit with genarater
        pspnet.fit_generator(generator=zip(xtr, ytr),
                steps_per_epoch=args.epoch_steps,
                epochs=args.n_epochs,
                validation_data=zip(xval, yval),
                validation_steps=args.val_steps,
                callbacks=[cp_cb, es_cb, tb_cb])

    # save model
    with open("./pretrained_mask/LIP_SegUNet_mask.json", "w") as json_file:
        json_file.write(json.dumps(json.loads(segunet.to_json()), indent=2))
    print("save json model done...")


def ff( x, y, bs ):
   rs = []
   for i in range(len(x)):
     for j in range(bs):
       

