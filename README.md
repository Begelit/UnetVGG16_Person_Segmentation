# UnetVGG16_Person_Segmentation
![](https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/demo/collage_gif.gif)
## 1 Creating and training CNN model based on Unet architecture with VGG16 encoder
The following sections will describe the actions performed in this notebook:
https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/Jupyter%20Notebooks/unet-keras-person-cocodataset2017.ipynb
More detailed information about the session of this section, input / output files can be obtained at the link:
https://www.kaggle.com/code/dimalevch/unet-keras-person-cocodataset2017/data
### 1.1 Dataset
In my opinion, COCO Dataset 2017 was the best choice for the human segmentation task because it has a configuration file managed by COCO API. In addition, the dataset has hundreds of thousands of the most varied images and several dozen classes, which allows neglecting augmentation.
#### 1.1.1 Download and inport COCO API
```python
!pip install -q git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
from pycocotools.coco import COCO
```
#### 1.1.2 Get list of images index in JSON annotations file
This process is implemented by the function ```getImgsNamesList()``` described in the notebook, the link of which is at the beginning of the section.
Calls to this function look like this:
```python
catIds_train, imgIds_train, coco_train = getImgsNamesList(dataType='train2017',classNames='person')
catIds_val, imgIds_val, coco_val = getImgsNamesList(dataType='val2017',classNames='person')
```
Where the first variable stores the index of the class(In our case, the person's class index). 
The second variable stores the index of the image in the configurator, which allows you to return a dictionary with links to the image, for example:
```python
image_dict = coco_train.loadImgs(imgIds_train)[0]
print(image_dict)
```
Results:
```python
{'license': 2,
 'file_name': '000000262145.jpg',
 'coco_url': 'http://images.cocodataset.org/train2017/000000262145.jpg',
 'height': 427,
 'width': 640,
 'date_captured': '2013-11-20 02:07:55',
 'flickr_url': 'http://farm8.staticflickr.com/7187/6967031859_5f08387bde_z.jpg',
 'id': 262145}
```
#### 1.1.3. Batch data generator
Due to the limited RAM memory of computing devices, to train the model, it is necessary to divide the data sets into small packets, which will be sequentially fed to the input of our CNN.
The class ``` DataGenerator()``` implements a sequential passage through the configuration of the coco dataset, extracts a set of image arrays and forms their binary masks. 
Let's try to create an instance of the class, call the generator function and get the images.
```python
img_path_val = '../input/coco-2017-dataset/coco2017/val2017'
img_path_train = '../input/coco-2017-dataset/coco2017/train2017'
train_gen = DataGenerator(catIds_train,imgIds_train,coco_train,32,img_path_train)
val_gen = DataGenerator(catIds_val,imgIds_val,coco_val,32,img_path_val)
x, y = train_gen.__getitem__(1)
import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(array_to_img(x[1]))
plt.imshow(y[1])
```
As a result, we get the image-mask correspondence:

![](https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/demo/DataGeneratorResult.png)

### 1.2 Architecture of the Unet Convolutional Neural Network model
#### 1.2.1 Layers
Model building is achieved by creating and calling custom functions: ```conv_block()```,```define_decoder()```,```vgg16_unet()```.

The constructed layers of the model can be visualized as follows:

![](https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/demo/VisualModel_1.png)

More detailed information about the layers, their size and purpose in the diagram below:

![](https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/demo/model-unet.png)

#### 1.2.2 Metrics
For assess the quality of the trained model, the Jaccard index was used.

![https://upload.wikimedia.org/wikipedia/commons/c/c7/Intersection_over_Union_-_visual_equation.png](https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/demo/Intersection_over_Union_-_visual_equation.png)

Borrowed from source:https://upload.wikimedia.org/wikipedia/commons/c/c7/Intersection_over_Union_-_visual_equation.png

The functions of this metric are implemented as follows:

```python
def jacard_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)
    
def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)
```
#### 1.2.3 Compile and train the model
Lets compile the model model:

```python
model = vgg16_unet(input_shape = (256,256,3))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=[jacard_coef_loss], metrics=['accuracy', jacard_coef])
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint('./model_epoch_{epoch:00d}', save_best_only= False)
callbacks = [checkpoint,reduce_lr]
```

And start to train the model:

```python
epochs = 15
model_history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
```

Training process:

![](https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/demo/TrainProcess.PNG)

## 2 Predicting an image mask from a test dataset
Model loading methods and image mask prediction are described in this notebook: https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/Jupyter%20Notebooks/predict-coco-unet-epoch21.ipynb

To access the details of the session, you can go to it at the link: https://www.kaggle.com/code/dimalevch/predict-coco-unet-epoch21

### 2.1 Load model
For successful loading of the model, you must specify inside the method ```tf.keras.models.load_model()``` customized functions ```jacard_coef()``` and ```jacard_coef_loss()``` which are used in assessing the quality of model training.

The model load function call would be as follows:
```python
model = tf.keras.models.load_model('../input/model-coco-unet-segm-epoch-21/model_coco_epoch_21',
                                   custom_objects={'jacard_coef':jacard_coef,'jacard_coef_loss':jacard_coef_loss})
```
### 2.2 Data Generator
The function ```dataGenerator_test()``` implements a random receipt of an image from a test data set into an array with its further normalization.

A function call that returns a batch of 15 images will look like this:

```python
path_images = '../input/coco-2017-dataset/coco2017/test2017'
for x in dataGenerator_test(15,path_images):
    break
```

### 2.3 Prediction and visualization of results

```python
import matplotlib.pyplot as plt
%matplotlib inline
for index in range(15):
    plt.figure(figsize=(8,16))
    pred = model.predict(x[index].reshape((1, 256, 256,3)))
    plt.subplot(1,2,1)
    plt.title('Image')
    plt.imshow(array_to_img(x[index]))
    plt.subplot(1,2,2)
    plt.title('Predicted Mask')
    plt.imshow(array_to_img(pred.reshape((256, 256, 1))))
```
![](https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/demo/PredictEpoch21.png)

## 3 Removing the background around a segmented object. Video Frame Prediction
### 3.1 Splitting a video file into frames
This process is demonstrated in this notebook:
https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/Jupyter%20Notebooks/video-to-frames.ipynb
And the corresponding session is available at the link:
https://www.kaggle.com/code/dimalevch/video-to-frames
### 3.2 Predict frames from video
https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/Jupyter%20Notebooks/predict-frames-from-video-unetsegm.ipynb
https://www.kaggle.com/code/dimalevch/predict-frames-from-video-unetsegm

As a result, we get an image of masks equal in number to frames from the video:

![](https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/demo/VideoToFramesPredict.png)

### 3.3 White background behind segmented object
https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/Jupyter%20Notebooks/create-white-background-with-mask.ipynb
https://www.kaggle.com/code/dimalevch/create-white-background-with-mask
![](https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/demo/collageFrame.PNG)

### 3.4 Create video/gif from collage frames
https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/Jupyter%20Notebooks/create-gif-and-video-collages-unetseg.ipynb
https://www.kaggle.com/code/dimalevch/create-gif-and-video-collages-unetseg
![](https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/demo/collage_gif.gif)


