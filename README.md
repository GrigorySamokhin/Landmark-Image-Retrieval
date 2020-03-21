#  Landmark Image Classification with K Nearest Neighbours & CBIR

A program uses the content of an image, to search for the most similar images in a database. In order to find the closest match, the system must use an algorithm to efficiently find key "descriptors" for an image that can be used to compare to the descriptors of images in the database.

##   Project Objectives
>  Extracted keypoint detectors with ResNet pre-trained model from dataset

>  Accepted a query image

>  Using K Nearest Neighbours Classification

### Data
For this project i choose [Paris6k](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) dataset, but i cropped it to 50 images per class for KNN, photos stored in **paris** directory. 
Testing data is set of photos from google stored in **paris_test** directory. But you can generate own testing directory with following command (default size of test_dataset: 10%):

    python3 main.py --mode create_test_dataset --dataset $PATH
    Dataset structure: dataset/ class_1/
                    class_2/
                    ...
                    class_n/

### Model
ResNet have been trained to recognise 1000 different categories from the ImageNet database. However we can also use them to extract a feature vector (a list of 2048 floating point values) of the models internal representation of a category. The computed vectors from dataset are stored in metadata/ directory in .cpickle format. The challenge is to handle with viewpoint variation, illumination conditions, scale variation, deformation. 

Another implementation feature detection: 
- Harris Corner Detector
- SIFT (Scale-Invariant Feature Transform)
- SURF (Speeded-Up Robust Features)
- FAST (Features from Accelerated Segment Test)
- BRIEF (Binary Robust Independent Elementary Features)
- ORB (Oriented FAST and Rotated BRIEF)
- LBP (Local Binary Patterns)
- Use another pre-trained NN

Taking the problem of feature extraction, we can compare the two types of algorithms for computer vision:

The traditional approach is to use well-established CV techniques such as feature descriptors (SIFT, SURF, BRIEF, etc.). The difficulty with this traditional approach is that it is necessary to choose which features are important in each given image. As the number of classes to classify increases, feature extraction becomes more and more cumbersome.

The Deep Learning approach to use pre-trained Neural Network or write it from scratch. The advantage of using pre-trained models is that they have learned how to detect generic features from photographs, given that they were trained on more than 1,000,000 images for 1,000 categories.

### Prediction

Prediction based on KNN classification. The model calculate the distance (using L1 norm or L2 norm distances) between query image and every image in dataset and take n (DEPTH in code) images.
After computing we need to load query image and get predictions by following command:

    python3 main.py --mode predict --dataset paris --query paris_test/eiffel_tower/10-projets-de-voyage-a-PARIS.jpg
   ![enter image description here](https://psv4.userapi.com/c856536/u21543301/docs/d11/ee72ca94059c/10-projets-de-voyage-a-PARIS.jpg?extra=x7wrxiN9q7mgoiEUABadee7AXgLGsG2rrNM5YdIziRrDns6gJTXj8awrEDrTeqzCRYu_G_SZu4EsSxBo6j-EaFXngGFjaNqd6wlOKgO0BvKdaKE4qhfilPdLPxvdCAfeI0_VOR3k_YCBSksEB89A-Ro)
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

### Content Based Image Retrieval 
You can print nearest photo in dataset by command:

    python3 --mode cbir --dataset paris --query paris_test/eiffel_tower/paris-1513078302.jpg
Output:
![](https://sun9-2.userapi.com/c855124/v855124614/20ffb9/ArmXsj9rbcw.jpg)

 ![enter image description here](https://sun9-50.userapi.com/c855124/v855124614/20ffc0/oV-3ByAj4T4.jpg)

![enter image description here](https://sun9-65.userapi.com/c855124/v855124614/20ffc7/1KkTzJYkAQ0.jpg)

![enter image description here](https://sun9-57.userapi.com/c855124/v855124614/20ffd5/xHQuFcNKuR8.jpg)

### Metrcis
**MAP (Mean Average Precision)**

    python3 --mode map --dataset paris
  Output

    Mode type: map.
    Distance type: L1.
    GPU available: False.
    Dataset directory: paris.
    Computed feature vectors of dataset: True.
    
    MAP based on first 3 vectors
    Class: moulinrouge, MAP: 1.0
    Class: eiffel_tower, MAP: 0.94
    Class: defense, MAP: 0.99
    Class: louvre, MAP: 0.93
    Class: invalides, MAP: 0.98
    MMAP:  0.968888888888889

  **Accuracy**

    python3 --mode accuracy --dataset paris --test_dataset paris_test
Output

    Mode type: accuracy.
    Distance type: L1.
    GPU available: False.
    Dataset directory: paris.
    Computed feature vectors of dataset: True.
        
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
                   [--query QUERY] [--mode MODE] [--distance DISTANCE]
    
    Image Classification based on ResNet pre-trained model and L1 norm
    
    optional arguments:
      -h, --help            show this help message and exit
      --dataset DATASET     Path to your dataset
      --test_dataset TEST_DATASET
                            Path to your test dataset, work if mode: accuracy
      --query QUERY         Path to your query: example.jpg
      --mode MODE           train: Extract feature vectors from given dataset,
                            predict: Predict a class of query map: Calculate an
                            average precision of dataset accuracy: Calculate
                            accuracy of prediction on testing dataset
                            create_test_dataset: Creating a testing dataset from
                            dataset cbir: Print 3 most similar photo for query
      --distance DISTANCE   L1: Manhattan Distance L2: Euclidean distance



### References

 - [https://medium.com/swlh/image-classification-with-k-nearest-neighbours-51b3a289280](https://medium.com/swlh/image-classification-with-k-nearest-neighbours-51b3a289280)
- [https://medium.com/analytics-vidhya/introduction-to-feature-detection-and-matching-65e27179885d](https://medium.com/analytics-vidhya/introduction-to-feature-detection-and-matching-65e27179885d)
- [https://www.pyimagesearch.com/2014/02/03/building-an-image-search-engine-defining-your-image-descriptor-step-1-of-4/](https://www.pyimagesearch.com/2014/02/03/building-an-image-search-engine-defining-your-image-descriptor-step-1-of-4/)
-  [https://towardsdatascience.com/cnn-application-on-structured-data-automated-feature-extraction-8f2cd28d9a7e](https://towardsdatascience.com/cnn-application-on-structured-data-automated-feature-extraction-8f2cd28d9a7e)
- [https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/](https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/)