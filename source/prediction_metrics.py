import numpy as np
from collections import Counter
import os
import random
from shutil import move
import matplotlib.pyplot as plt

DEPTH = 3
TEST_DATASET_SIZE = .1

def count_distance(vector_1, vector_2, distance_type):
    '''
    :param vector_1:
    :param vector_2:
    :param distance_type: 'L1' Manhattan distance or 'L2' Euclidean distance

    Counting a distance between two vectors
    '''
    # Manhattan distance
    if distance_type == 'L1':
        return np.sum(np.absolute(vector_1 - vector_2))

    # Euclidean distance
    elif distance_type == 'L2':
        return np.sum((vector_1 - vector_2)**2)

def predict_class(query, vectors, distance, mode=0):
    '''
    :param query: Query Image dict with {'image_path': Path to image,
                                    'class_image': Class of an image,
                                    'feature_vector': Feature vector of image}
    :param vectors: List with query-like type dict of whole dataset
    :param distance: Type of distance
    :param mode:    0 - Print result of prediction
                    1 - Return result of prediction
                    2 - Return list of DEPTH predictions

    Computing distance between query image vector and all vectors in dataset.
    Sort result list. Take n(DEPTH) best distances and return most frequently class.
    '''
    result_list = []

    # Count distance between query and dataset vectors
    for temp_vector in vectors:
        distance_len = count_distance(temp_vector['feature_vector'], query['feature_vector'], distance)
        result_list.append({
            'class': temp_vector['class_image'],
            'distance': distance_len,
            'image_path': temp_vector['image_path']
        })

    # Sorting list by distance
    result_list = sorted(result_list, key=lambda x: x['distance'])

    # Get best distance
    result_best_distance = min(result_list, key=lambda x: x['distance'])

    # Get most frequent class in (DEPTH) first classes
    res_count = Counter([x['class'] for x in result_list][:DEPTH])
    res = min(res_count.items(), key=lambda x: (-x[1], x[0]))[0]

    if mode == 2:
        return result_list[:DEPTH]

    if mode == 1:
        return res

    # Check similarity with dataset
    if result_best_distance['distance'] < 0.5:
        print('\nPredicted class for query image: {}.'.format(res))
    else:
        print('\nQuery is not similar to any class.')

def get_ap(label, results):
    '''
    :param label: True class of image
    :param results: DEPTH-samples of best distance predicted classes

    Counting an average precision
    :return:
    '''
    precision = []
    hit = 0

    # Calculate precision
    for i, result in enumerate(results):

        if result['class'] == label:
            hit += 1

        # if best distance not the same class like label
        if hit == 0:
            return 0.
        precision.append(hit / (i + 1.))

    return np.mean(precision)


def calculate_ap(query, vectors, distance):
    '''
    :param query: Query Image dict with {'image_path': Path to image,
                                    'class_image': Class of an image,
                                    'feature_vector': Feature vector of image}
    :param vectors: List with query-like type dict of whole dataset
    :param distance: distance: Type of distance

    Calculating an average precision for query
    :return:
    '''
    results = []

    # Create result list with distances between query and vectors in dataset
    for temp in vectors:
        if query['image_path'] == temp['image_path']:
            continue
        distance_len = count_distance(query['feature_vector'], temp['feature_vector'], distance)
        results.append({'class': temp['class_image'],
                        'distance': distance_len})

    # Sorting list by distance
    results = sorted(results, key=lambda x: x['distance'])

    # Take DEPTH samples
    results = results[:DEPTH]

    # Calculate AP
    ap = get_ap(query['class_image'], results)
    return ap

def calculate_map_metric(dataset, FeatureExtractor, distance):
    '''
    :param dataset: Object of DataSet with images and classes
    :param FeatureExtractor: Object of FeatureExtractor
    :param distance: Type of distance


    Compute all Images from dataset into feature vectors.
    Calculating an Average Precision for each image in class.
    Calculating Mean Average Precision for each class and Mean MAP
    '''

    # Get list of labels
    labels = dataset.get_labels()
    ret = {c: [] for c in labels}
    mean_ap = []

    # Create object to create feature vectors from photos
    extractor = FeatureExtractor(dataset.dataset_dir)

    # Create vectors
    vectors = extractor.feature_vectors(dataset)

    # Calculate Average Precision for each image and store it for each class
    for temp_vector in vectors:
        ap = calculate_ap(temp_vector, vectors, distance)
        ret[temp_vector['class_image']].append(ap)

    # Calculate MAP and MMAP
    print('\nMAP based on first {} vectors'.format(DEPTH))
    for class_temp, ap_temp in ret.items():
        map = np.mean(ap_temp)
        mean_ap.append(map)
        print("Class: {}, MAP: {}".format(class_temp, round(map, 2)))
    print("MMAP: ", np.mean(mean_ap))

def calculate_accuracy_metric(dataset, test_dataset, FeatureExtractor, distance):
    '''
    :param dataset: Object of main DataSet with images and classes
    :param test_dataset: Object of test DataSet with images and classes
    :param FeatureExtractor: Object of FeatureExtractor
    :param distance: Type of distance

    Compute all Images from dataset into feature vectors.
    Going through test_dataset directory and make prediction for every image.
    When calculating an accuracy for each class
    '''

    # Create object to create feature vectors from photos
    extractor = FeatureExtractor(dataset.dataset_dir)

    # Create vectors
    vectors = extractor.feature_vectors(dataset)

    all_result = []

    # For each image in dataset_test make predictions and check accuracy for each class
    for class_path in os.listdir(test_dataset):
        right_pred = 0
        for query_path in os.listdir(os.path.join(test_dataset, class_path)):

            # Create feature vector
            query = extractor.compute_query_vector(os.path.join(test_dataset, class_path, query_path))

            # Make prediction and check it
            class_pred = predict_class(query, vectors, distance, 1)
            if class_pred == class_path:
                    right_pred += 1
        accuracy_class = right_pred / len(os.listdir(os.path.join(test_dataset, class_path)))
        all_result.append({'accuracy': accuracy_class,
                           'class': class_path})
    print('\n')
    for temp_list in all_result:
        print('Class: {}, accuracy: {}.'.format(temp_list['class'], round(temp_list['accuracy'], 2)))

def create_test_class(source, class_dir):
    '''

    :param source: Name of dataset directory
    :param class_dir: Name of class directory to split

    Move n (TEST_DATASET_SIZE %) from source/class_dir directory
    to source_test/class_dir
    :return:
    '''

    files = []
    test_dir = os.path.join(source + '_test', class_dir)

    # Check files im dataset
    for filename in os.listdir(os.path.join(source, class_dir)):
        file = os.path.join(source, class_dir, filename)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    # Compute len and shuffle dataset
    training_length = int(len(files) * (1 - TEST_DATASET_SIZE))
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    testing_set = shuffled_set[:testing_length]

    # move files from dataset to test_dataset
    for filename in testing_set:
        this_file = os.path.join(source, class_dir, filename)
        destination = os.path.join(test_dir, filename)
        move(this_file, destination)

def create_test_dataset(dataset_dir):
    '''
    :param dataset_dir: Path to dataset

    Creating a dataset_test, size of dataset_test is TEST_DATASET_SIZE %
    '''
    test_dir = dataset_dir + '_test'

    # Create dataset_test directoru
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # Move Images for each directory in dataset
    for class_dir in os.listdir(dataset_dir):
        if not os.path.exists(os.path.join(test_dir, class_dir)):
            os.mkdir(os.path.join(test_dir, class_dir))
        create_test_class(dataset_dir, class_dir)
    print('Create {}.'.format(test_dir))

def print_nearest_photo(query, vectors, distance):
    '''
    :param query: Query Image dict with {'image_path': Path to image,
                                    'class_image': Class of an image,
                                    'feature_vector': Feature vector of image}
    :param vectors: List with query-like type dict of whole dataset
    :param distance: Type of distance
    :param mode:    0 - Print result of prediction
                    1 - Return result of prediction

    Calculate nearest photos and prints nearest DEPTH samples
    '''
    results = predict_class(query, vectors, distance, mode=2)

    plt.imshow(np.array(plt.imread(query['image_path']), dtype=int))
    plt.title('Query Image')
    plt.show()

    for i, vectors in enumerate(results):
        image_path = vectors['image_path']
        image = np.array(plt.imread(image_path), dtype=int)
        plt.title('Result {}'.format(i))
        plt.imshow(image)
        plt.show()

