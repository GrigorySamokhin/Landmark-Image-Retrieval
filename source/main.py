import torch
import argparse
from model import FeatureExtractor
from prediction_metrics import calculate_map_metric, predict_class, \
    calculate_accuracy_metric, create_test_dataset, print_nearest_photo
from dataset import DataSet


USE_GPU = torch.cuda.is_available()
MODEL_URL_RESNET_152 = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'

parser = argparse.ArgumentParser(description='Image Classification based on ResNet pre-trained model and L1 norm')
parser.add_argument('--dataset', type=str, help='Path to your dataset', required=True)
parser.add_argument('--test_dataset', type=str, help='Path to your test dataset, work if mode: accuracy')
parser.add_argument('--query', type=str, help='Path to your query: example.jpg')
parser.add_argument('--mode', type=str, help='train: Extract feature vectors from given dataset, \n'
                                             'predict: Predict a class of query \n'
                                             'map: Calculate an average precision of dataset\n'
                                             'accuracy: Calculate accuracy of prediction on testing dataset\n'
                                             'create_test_dataset: Creating a testing dataset from dataset\n'
                                             'cbir: Print 3 most similar photo for query')
parser.add_argument('--distance', type=str, help='L1: Manhattan Distance \n'
                                            'L2: Euclidean distance', required=False, default='L1')
args = parser.parse_args()

def print_defenitions():
    """
    Print mode of execution, distance type and available GPU
    """
    print('Mode type: {}.'.format(args.mode))
    print('Distance type: {}.'.format(args.distance))
    if USE_GPU:
        print('GPU available: True.')
    else:
        print('GPU available: False.')

if __name__ == "__main__":
    print_defenitions()
    dataset = DataSet(args.dataset)
    if args.mode == 'train':
        feat_ex = FeatureExtractor(dataset.dataset_dir)
        feat_ex.feature_vectors(dataset)
    elif args.mode == 'predict':
        extractor = FeatureExtractor(dataset.dataset_dir)
        vectors = extractor.feature_vectors(dataset)
        predict_class(extractor.compute_query_vector(args.query), vectors, args.distance)
    elif args.mode == 'map':
        calculate_map_metric(dataset, FeatureExtractor, args.distance)
    elif args.mode == 'accuracy':
        calculate_accuracy_metric(dataset, args.test_dataset, FeatureExtractor, args.distance)
    elif args.mode == 'create_test_dataset':
        create_test_dataset(dataset.dataset_dir)
    elif args.mode == 'cbir':
        extractor = FeatureExtractor(dataset.dataset_dir)
        vectors = extractor.feature_vectors(dataset)
        print_nearest_photo(extractor.compute_query_vector(args.query), vectors, args.distance)

