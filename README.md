# Cat-Face-Verification
A model using triplet loss to solve the task of cat face verification
This is a project that is similar to one the paper of it is published at <a href="http://cs230.stanford.edu/projects_fall_2019/reports/26251543.pdf">Pet Cat Face Verification and Identification</a>.
## Dataset
However, due to the fact that I could not access to the API stipulated in the paper, I have improvised a way to obtain cat images online (tHrough Instagram accounts that post about cat). There are 32 cats in my dataset, each has at least 10 images. The next step is to obtain image of their faces only. I used Yolov5 for this task. By labelling by hand with the help of Label Studio about 100 images of cat face, I created a training dataset big enough for training on Yolov5. The result is quite satisfactory. 

## Model
The model I use in this project is a simpler replicate of Resnet. (Insert model).This model outputs an 32 dimensional embedding feature vector given 224*224*3 image. The last layer of the model is a Lambda one that L2 normalize the embedding vector feature. 
## Triplet Loss
The approach I use for building loss function is online tripling. This approach is executed spetacularly in <a href="https://omoindrot.github.io/triplet-loss">this artice</a>. However, in my implementation of this loss function, I came to realize that without a customized regularization term, the model tend to push all the embeddings to zero, causing the loss to be margin. This issue is also recorded in the paper. As a result, I incorporate the regularization term into the loss function. This customized regularization punishes the model when the anchor vector and the negative vector are similar.
The main hyperparameters of the loss function is: 
1. Margin: indicate how far we expect the gap between (anchor_embedding-negative_embedding) and (anchor_embedding-positive_embedding)
2. Alpha: indicate how influential the regularization term is

## Training

## Testing
