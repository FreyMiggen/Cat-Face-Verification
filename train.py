import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Lambda, BatchNormalization
from tensorflow.keras.utils import plot_model
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from utils import *
import argparse

def init_model(pretrained_model_path='none'):
    if pretrained_model_path=='none':
        
        resnet_base=tf.keras.applications.ResNet50V2(include_top=False,
                                                    weights='imagenet',
                                                    input_shape=(224,224,3))
       

        resnet_base.trainable=True
        layers=resnet_base.layers
        for layer in layers[:98]:
            layer.trainable=False
        
        inputs=Input(shape=(224,224,3))
        x=resnet_base(inputs)
        x=GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dropout(0.25)(x)
        x= Dense(512,activation='relu')(x)
        x = Dropout(0.25)(x)
        x= Dense(128,activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(32, use_bias=False)(x)
        outputs = Lambda(lambda x: tf.math.l2_normalize(x,axis=-1))(x)
        model = tf.keras.Model(inputs,outputs)
    else:
        model=tf.keras.models.load_model(pretrained_model_path)
        print("Model initiated successfully!")


    return model

    

def build_dataset(data_dir):
    
    dataset=tf.keras.preprocessing.image_dataset_from_directory(directory=data_dir,
                                                           image_size=(224,224),
                                                           batch_size=64,
                                                           color_mode='rgb',
                                                           shuffle=True)

    dataset=dataset.map(process_img)
    return dataset

def train(datadir,margin=0.2,alpha=0,num_epochs=40,pretrained_model_path='none'):
    model=init_model(pretrained_model_path)
    dataset=build_dataset(datadir)
    print('build dataset finished')
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    for epoch in range(num_epochs):
        running_loss=[]
        running_acc=[]

        for x_batch,y_batch in dataset:
            
            with tf.GradientTape() as tape:
                embeddings=model(x_batch,training=True)
                loss=batch_all_triplet_loss(y_batch,embeddings,margin,alpha)
                acc=accuracy_triplet(y_batch,embeddings,margin)
            gradients=tape.gradient(loss,model.trainable_weights)
            optimizer.apply_gradients(zip(gradients,model.trainable_weights))
            running_loss.append(loss)
            running_acc.append(acc)
        #calculate mean loss
        print(f'Loss at epoch {epoch}: ',np.mean(running_loss))
        print(f'Accuracy at epoch {epoch}: ',np.mean(running_acc))

   
    count=len(os.listdir('run'))
    os.makedirs(f'run/{count}/checkup')
    model.save(f'run/{count}/model.h5')
    
    foldir=[os.path.join(datadir,filename) for filename in os.listdir(datadir)]
    for sub_dir in foldir:
        saved_fol=f'run/{count}/checkup'
        select_anchor_vects(sub_dir,saved_fol,model)
    print(f'Embedding value of the new model for the database is stored in run/{count}/checkup')
    
    
def Main():
    parser=argparse.ArgumentParser()
    parser.add_argument('datadir',help='The path of the dataset folder')
    parser.add_argument('-a','--alpha',default=0,type=float)
    parser.add_argument('-m','--margin',default=0.2,type=float)
    parser.add_argument('-e','--epochs',default=40,type=int)
    parser.add_argument('-p','--pretrained_model',help='the path to the pretrained model',default='none')
    args=parser.parse_args()

    train(args.datadir,args.margin,args.alpha,args.epochs,args.pretrained_model)

if __name__=='__main__':
    Main()

