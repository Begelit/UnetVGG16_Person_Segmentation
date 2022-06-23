# UnetVGG16_Person_Segmentation
![](https://github.com/Begelit/UnetVGG16_Person_Segmentation/blob/main/demo/collage_gif.gif)
## 1 Creating and training CNN model based on Unet architecture with VGG16 encoder.
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
#### 1.1.2 Get list of images index in JSON annotations file.
This process is implemented by the function 
```python 
getImgsNamesList(dataDir='../input/coco-2017-dataset/coco2017/annotations',dataType=str(),classNames=str())
```
Described in the notebook, the link of which is at the beginning of the section.
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
#### 1.1.3. Batch data generator.
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

### 1.2 Architecture of the Unet Convolutional Neural Network model.
