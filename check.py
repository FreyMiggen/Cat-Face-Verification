import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
#import os
import argparse
import random

def processed_for_test(image_path,model):
    img = tf.keras.utils.load_img(image_path)
    img = tf.keras.utils.img_to_array(img)
    img=tf.image.resize(img,(224,224))
    
    input_arr=tf.keras.applications.resnet_v2.preprocess_input(img)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    return img/255.0,predictions

def test_result(filename,result):
    #result is an embedding vector of shape (1,32)
    # return False if 2 out of 3 vector return result>0.4
    anchors=np.loadtxt(filename,delimiter=',')
    dist=anchors-result
    dist=np.square(dist)
    dist=np.sum(dist,axis=1)
    dist=np.sqrt(dist)
    dist=np.less(dist,0.5)
    if np.sum(dist)>=2:
        return True
    else:
        return False

def detect(img_path,model_version='00'):
    if model_version in ['00','05','02']:
        model_path=f'resnet/resnet_50v2_emb_32_margin_02_alpha_{model_version}_40_epoch.h5'
        filedirs=[os.path.join(f'checkup/alpha{model_version}',filename) for filename in os.listdir(f'checkup/alpha{model_version}')]
    else:
        model_path=os.path.normpath(model_version)
        sub_dir='\\'.join(model_path.split('\\')[:-1])
        sub_dir=os.path.join(sub_dir,'checkup')
        filedirs=[os.path.join(sub_dir,filename) for filename in os.listdir(sub_dir)]

    model=tf.keras.models.load_model(model_path)
    img,test_emb=processed_for_test(img_path,model)
    for fol in filedirs:
        if test_result(fol,test_emb)==1:
            fol=os.path.normpath(fol)
            fname=fol.split('\\')[-1].split('.')[0].split('_')[-1]
            fol=os.path.join('0',fname)
            return fol
    return False

def Main():
    parser=argparse.ArgumentParser()
    parser.add_argument('img_path',help='Add the image path of the cat image you want to test')
    parser.add_argument('-m','--model_version',help='If you dont want to use the default model, declare the model version you want to use \n option: 00, 03,05',
    default='00')
    
    args=parser.parse_args()
    result=detect(args.img_path,args.model_version)
    if result:
  
        print(f'The cat is already stored in the database in {result}')
        img_list=[os.path.join(result,fname) for fname in os.listdir(result)]
        size=len(img_list)-1
        user_ans=input('Would you like to see another image of the cat?[y,n]: ')
        if user_ans=='y':
            ind=random.randint(0,size)
            img=Image.open(img_list[ind])
            img.show()
    else:
        print('The cat is not yet recorded in our database!')

    
    

if __name__=='__main__':
    Main()
