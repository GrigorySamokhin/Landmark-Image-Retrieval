#  Landmark Image Classification with K Nearest Neighbours | CBIR | VLAD + SIFT

A program uses the content of an image, to search for the most similar images in a database. In order to find the closest match, the system must use an algorithm to efficiently find key "descriptors" for an image that can be used to compare to the descriptors of images in the database.

##   Project Objectives
>  Extract feature using ResNet (like feature extractor) or SIFT + VLAD

>  Accepted a query image

>  Using KNN

## Data
For this project i chose [Paris6k](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) dataset. The original 
dataset was cropped to 150 images per class (5 classes, only representative images were left, all classes of the same size). 


Link to the dataset used in this project: 
[Dataset on Google Drive](https://drive.google.com/file/d/1kZoFNMNLW-q8_aobtpZlzyIO0rs9cFNX/view?usp=sharing).


But you use your own data, you can create own test dataset with following command (default size of test_dataset: 10%):

    python3 main.py --mode create_test_dataset --dataset $PATH
    Dataset structure: dataset/ class_1/
                    class_2/
                    ...
                    class_n/
                    
Two test datasets will be used to calculate the accuracy of our models:

- 10 photos per each class pre-selected from the main dataset. Included in link mentioned above.

![](https://sun9-31.userapi.com/c857624/v857624652/1cb6b5/kfWSKpW-jIE.jpg)

- 5 photo per each class from Google. 
[Link](https://drive.google.com/file/d/1-qz0SZL_FOf4nTSMWb7-oXud1ZqCT4no/view?usp=sharing)

![](https://sun9-27.userapi.com/c857624/v857624652/1cb6be/O2YudrDe3vY.jpg) 

As you can see, I tried to choose photos where the object for identification was not 
the size of the entire picture, and the photos had a lot of background.

## Two approaches
ResNet have been trained to recognise 1000 different categories from the ImageNet database. However we can also use them to extract a feature vector (a list of 2048 floating point values) of the models internal representation of a category. The computed vectors from dataset are stored in metadata/ directory in .cpickle format. The challenge is to handle with viewpoint variation, illumination conditions, scale variation, deformation. 

Another implementation feature detection: 
- Harris Corner Detector
- SIFT (Scale-Invariant Feature Transform)
- SURF (Speeded-Up Robust Features)
- FAST (Features from Accelerated Segment Test)
- BRIEF (Binary Robust Independent Elementary Features)
- ORB (Oriented FAST and Rotated BRIEF)
- LBP (Local Binary Patterns)
- ...

Taking the problem of feature extraction, we can compare the two types of algorithms for computer vision:

The traditional approach is to use well-established CV techniques such as feature descriptors (SIFT, SURF, BRIEF, etc.). The difficulty with this traditional approach is that it is necessary to choose which features are important in each given image. As the number of classes to classify increases, feature extraction becomes more and more cumbersome.

The Deep Learning approach to use pre-trained Neural Network or write it from scratch. The advantage of using pre-trained models is that they have learned how to detect generic features from photographs, given that they were trained on more than 1,000,000 images for 1,000 categories.

### Comparison

##### SIFT approach

Scale-Invariant Feature Transform (SIFT) is commonly used to describe local regions from
an image in a scale and rotational invariant way. Most of the time, SIFT refers to a two-step process
including keypoints detection (using for example the Difference of Gaussian (DoG) method) and
computation of SIFT descriptors around these keypoints. While the computed SIFT local descriptors are 
informative on their own, they do not provide a compact representation of the image. Since the number 
of extracted descriptors may vary between images, direct comparison between them is not possible. 
In our case we using VLAD (Vector of Locally Aggregated Descriptors). Each descriptor is assigned to 
its nearest cluster, and then, for each cluster, we consider the sum of differences between assigned 
descriptors and the centroid of the cluster in the SIFT 128-dimensional space.

##### Deep Learning approach
In this case we have several training strategies can be considered, depending on the application and the database:
	
- Full Training: consists in training both the convolution and the classification part of the network. 
Full training is more appropriate for databases of sufficient size or simple applications.

- Fine-Tuning: consists in training only part of the convolutional layers of a pre-trained network 
(trained on a huge database such as Imagenet). In this transfer learning method, shallow and general 
layers are kept frozen while deep layers are re-trained with a very small learning rate to learn features 
specific to the database. Fine-tuning generally works well for small databases.

- Feature extractor approach is a more direct case of transfer learning. It also uses a pre-trained network 
but directly treats it as a multi-usage generic feature extractor. No training is done on the convolutional 
part and feature map extraction can be performed at any level of the network. Based on these extracted features, 
any classifier could be used to obtain the final decision.

In our case, i chose the feature extractor approach in ResNet as the comparison basis with SIFT features, since 
it is an adapted solution for small datasets.

## Results 
#### MAP

Here are the conclusions for the dataset used (described in the date section). 
Two metrics were selected for evaluation, MAP and ACCURACY for the entire dataset. 
As we can see, due to the fact that garbage was removed from the original dataset and all the photos 
displayed the object belonging to their class, we were able to get quite high indicators (graph below). 

As we might expect, the VLAD + SIFT approach lags a bit behind ResNet, since there are many photos in the 
task of determining sequences, where the object is not in the center of the photo and SIFT recognizes many 
third-party local points. Resnet doesn't have this problem because its weights have been trained 
to highlight the main objects in the photo.

![](https://sun9-3.userapi.com/c205616/v205616722/dee48/Xxb6MaOtVsA.jpg)


#### Accuracy

With accuracy, things are similar, and we received high marks. 
If you look at the test photos (for which the accuracy was calculated), 
you will notice that the objects on them are located in the center of the photos, just like 
in the main database.  

The accuracy is calculated from pre-selected photos from the main dataset.

![](https://sun9-64.userapi.com/c205616/v205616722/dee4f/ZvUnxJGurJU.jpg)

But we can quickly check the stability of photos with noise in these two approaches, 
and use photos not from the original dataset. In this case, I used a set of 5 photos per 
class downloaded from Google (Mentioned in *Data*).

The accuracy is calculated from Google photos.

![](https://sun9-36.userapi.com/c857624/v857624652/1cb661/Pyw2_HXy1qE.jpg)

 Here we can see that the approach with ResNet is more robustness to different 
 angles and is resistant to identifying the main objects in the photo.


In General, such previous high rates are explained by the fact that the dataset was pre-processed 
from bad photos. With an increase in the number of noisy photos and unrepresentative photos in 
the dataset, the indicators would decrease, and you would have to use the classifier 
on Landmarks/non-Landmark photo. 

#### Diferent distance metrics

Since we use KNN in both cases let's compare two types of distances with different numbers 
of K neighbors and at the same time see how the map changes with their increase.

![](https://sun9-42.userapi.com/c205616/v205616722/dee3a/L4CBz1dbTbM.jpg)
![](https://sun9-16.userapi.com/c205616/v205616722/dee41/Wcr2BU99s94.jpg)

First, we can see that with increasing K of the nearest neighbors in both cases, 
MAP falls, this proves the intuitive fact that the nearest vectors by distance 
most likely show the same object. 

Second, we can see that ResNet's performance doesn't fall as much when we 
increase k, which shows that It is generally better at extracting features.

Third, we can see that at L2 distance SIFT+VLAD shows slightly better results, 
but in General we can assume that L1 shows better results on our dataset.

#### Diferent number of clusters (k-means) for VLAD-SIFT

![](https://sun9-35.userapi.com/c205616/v205616722/dee56/AR_DQ2ynPJs.jpg)

We can see accuracy are gradually increased with the number of clusters for VLAD-SIFT. 

## Prediction

Prediction based on KNN classification. The model calculate the distance (using L1 norm or L2 norm distances) between query image and every image in dataset and take n (DEPTH in code) images.
After computing we need to load query image and get predictions by following command:

    python3 main.py --mode predict --model resnet --dataset paris --query paris_test/eiffel_tower/10-projets-de-voyage-a-PARIS.jpg

  
![enter image description here](https://sun9-61.userapi.com/c857128/v857128691/1583c7/f9UDEKdM4wA.jpg)
  
  Output:

    Mode type: predict.
    Distance type: L1.
    GPU available: False.
    Dataset directory: paris.
    Computed feature vectors of dataset: False.
    Computing...
    
    Predicted class for query image: eiffel_tower. 
If we have pre-computed vectors we will see:

    Mode type: predict.
    Distance type: L1.
    GPU available: False.
    Dataset directory: paris.
    Computed feature vectors of dataset: True.
    
    Predicted class for query image: eiffel_tower.

On CPU all vectors in dataset **paris** with 50 images in 5 classes will computed in 30 min, in GPU 20 second.

## Content Based Image Retrieval 
You can print nearest photo in dataset by command:

    python3 main.py --mode cbir --model resnet --dataset paris --query paris_test/eiffel_tower/paris-1513078302.jpg

![](https://sun9-2.userapi.com/c855124/v855124614/20ffb9/ArmXsj9rbcw.jpg)
![](https://sun9-50.userapi.com/c855124/v855124614/20ffc0/oV-3ByAj4T4.jpg)
![](https://sun9-65.userapi.com/c855124/v855124614/20ffc7/1KkTzJYkAQ0.jpg)
![](https://sun9-57.userapi.com/c855124/v855124614/20ffd5/xHQuFcNKuR8.jpg)

##### VLAD + SIFT

We also can classify or retrieve images with [VLAD](https://lear.inrialpes.fr/pubs/2010/JDSP10/jegou_compactimagerepresentation.pdf) (Vector of Locally Aggregated Descriptors) and SIFT (Scale-invariant feature transform) approach.
To retieve images with VLAD + SIFT use:

    python3 main.py --mode cbir --model vlad --dataset paris --query paris_test/louvre/louvre.jpg


![](https://sun9-22.userapi.com/c205816/v205816415/b381c/VrNrlsBkzg4.jpg)
![](https://sun9-19.userapi.com/c205816/v205816415/b3823/Amux3NyoCL0.jpg)
![](https://sun9-49.userapi.com/c205816/v205816415/b382a/b1sHbqsFQ-M.jpg)
![](https://sun9-41.userapi.com/c205816/v205816415/b3831/KaRXn5yD1j0.jpg)

To classify query image use:

    python3 main.py --mode predict --model vlad --dataset paris --query paris_test/eiffel_tower/2019_05_30_73565_1559195283._large.jpg

Output:
    
    Predicted class for query image: eiffel_tower.
    
### Metrcis
**MAP (Mean Average Precision)**

Resnet:

    python3 main.py --metric map --model resnet --dataset paris
  Output
    
    MAP based on first 3 vectors
    Class: moulinrouge, MAP: 1.0
    Class: eiffel_tower, MAP: 0.94
    Class: defense, MAP: 0.99
    Class: louvre, MAP: 0.93
    Class: invalides, MAP: 0.98
    MMAP:  0.968888888888889

VLAD:

    python3 main.py --metric map --model vlad --dataset paris
Output:  
   
    MAP based on first 3 vectors
    Class: louvre, MAP: 0.79
    Class: defense, MAP: 0.93
    Class: eiffel_tower, MAP: 0.89
    Class: invalides, MAP: 0.93
    Class: moulinrouge, MAP: 0.99
    MMAP:  0.906
  **Accuracy**


    python3 main.py --metric accuracy --model resnet --dataset paris --test_dataset paris_test
Output
        
    Class: defense, accuracy: 0.8.
    Class: eiffel_tower, accuracy: 1.0.
    Class: invalides, accuracy: 1.0.
    Class: louvre, accuracy: 1.0.
    Class: moulinrouge, accuracy: 1.0.

### System Requirements
    Python 3.7 and packages:
            numpy       == 1.16.2
            pytorch     == 1.0.1
            opencv      == 3.4.1
            torchvision == 0.2.2
            matplotlib  == 3.0.3
            six         == 1.12.0


### Usage

    usage: main.py [-h] --dataset DATASET [--test_dataset TEST_DATASET]
                   [--query QUERY] [--model MODEL] [--mode MODE] [--metric METRIC]
                   [--distance DISTANCE]
    
    Image Classification based on ResNet pre-trained model and L1 norm
    
    optional arguments:
      -h, --help            show this help message and exit
      --dataset DATASET     Path to your dataset
      --test_dataset TEST_DATASET
                            Path to your test dataset, work if mode: accuracy
      --query QUERY         Path to your query: example.jpg
      --model MODEL         vlad: Predict class using VLAD + SIFT resnet: Pre-
                            trained ResNet
      --mode MODE           train: Extract feature vectors from given dataset,
                            predict: Predict a class of query create_test_dataset:
                            Creating a testing dataset from dataset cbir: Print 3
                            most similar photo for queryvlad: Predict class using
                            VLAD + SIFT
      --metric METRIC       map: Calculate an average precision of dataset
                            accuracy: Calculate accuracy of prediction on testing
                            dataset
      --distance DISTANCE   L1: Manhattan Distance L2: Euclidean distance




### References

 - [https://medium.com/swlh/image-classification-with-k-nearest-neighbours-51b3a289280](https://medium.com/swlh/image-classification-with-k-nearest-neighbours-51b3a289280)
- [https://medium.com/analytics-vidhya/introduction-to-feature-detection-and-matching-65e27179885d](https://medium.com/analytics-vidhya/introduction-to-feature-detection-and-matching-65e27179885d)
- [https://www.pyimagesearch.com/2014/02/03/building-an-image-search-engine-defining-your-image-descriptor-step-1-of-4/](https://www.pyimagesearch.com/2014/02/03/building-an-image-search-engine-defining-your-image-descriptor-step-1-of-4/)
-  [https://towardsdatascience.com/cnn-application-on-structured-data-automated-feature-extraction-8f2cd28d9a7e](https://towardsdatascience.com/cnn-application-on-structured-data-automated-feature-extraction-8f2cd28d9a7e)
- [https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/](https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/)