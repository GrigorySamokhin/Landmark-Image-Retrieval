import torch
import argparse
from model import FeatureExtractor
from prediction_metrics import *
from dataset import DataSet
from vlad import VladPrediction


USE_GPU = torch.cuda.is_available()
MODEL_URL_RESNET_152 = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'

parser = argparse.ArgumentParser(description='Image Classification based on ResNet pre-trained model and L1 norm')
parser.add_argument('--dataset', type=str, help='Path to your dataset', required=True)
parser.add_argument('--test_dataset', type=str, help='Path to your test dataset, work if mode: accuracy')
parser.add_argument('--query', type=str, help='Path to your query: example.jpg')
parser.add_argument('--model', type=str, help='vlad: Predict class using VLAD + SIFT \n'
                                                'resnet: Pre-trained ResNet')
parser.add_argument('--mode', type=str, help='train: Extract feature vectors from given dataset, \n'
                                                'predict: Predict a class of query \n'
                                                'create_test_dataset: Creating a testing dataset from dataset\n'
                                                'cbir: Print 3 most similar photo for query'
                                                'vlad: Predict class using VLAD + SIFT')
parser.add_argument('--metric', type=str, help='map: Calculate an average precision of dataset\n'
                                                'accuracy: Calculate accuracy of prediction on testing dataset\n')
parser.add_argument('--distance', type=str, help='L1: Manhattan Distance \n'
                                                'L2: Euclidean distance', required=False, default='L1')
args = parser.parse_args()

def print_defenitions():
    """
    Print mode of execution, distance type and available GPU
    """
    print('Model type: {}'.format(args.model))
    print('Mode type: {}.'.format(args.mode))
    print('Metric type: {}'.format(args.metric))
    print('Distance type: {}.'.format(args.distance))
    if USE_GPU:
        print('GPU available: True.')
    else:
        print('GPU available: False.')

if __name__ == "__main__":
    print_defenitions()

    dataset = DataSet(args.dataset)
    vlad_dataset = DataSet(args.dataset)

    if args.mode == 'train':
        feat_ex = FeatureExtractor(dataset.dataset_dir)
        feat_ex.feature_vectors(dataset)

    elif args.mode == 'predict':
        if args.model == 'resnet':
            extractor = FeatureExtractor(dataset.dataset_dir)
            vectors = extractor.feature_vectors(dataset)
            predict_class(extractor.compute_query_vector(args.query), vectors, args.distance)
        elif args.model == 'vlad':
            vlad_class = VladPrediction(vlad_dataset, args.dataset, args.query)
            vlad_vectors = vlad_class.vlad_prediction(vlad_dataset)

    elif args.metric == 'map':
        if args.model == 'resnet':
            calculate_map_metric(dataset, FeatureExtractor, args.distance)
        elif args.model == 'vlad':
            vlad_class = VladPrediction(vlad_dataset, args.dataset, args.query)
            calculate_map_vlad_metric(vlad_dataset, vlad_class)

    elif args.metric == 'accuracy':
        if args.model == 'resnet':
            calculate_accuracy_metric(dataset, args.test_dataset, FeatureExtractor, args.distance)
        elif args.model == 'vlad':
            vlad_class = VladPrediction(vlad_dataset, args.dataset, args.query)
            calculate_accuracy_metric_vlad(dataset, args.test_dataset, vlad_class)

    elif args.mode == 'cbir':
        if args.model == 'resnet':
            extractor = FeatureExtractor(dataset.dataset_dir)
            vectors = extractor.feature_vectors(dataset)
            print_nearest_photo(extractor.compute_query_vector(args.query), vectors, args.distance)
        elif args.model == 'vlad':
            vlad_class = VladPrediction(vlad_dataset, args.dataset, args.query)
            vlad_vectors = print_nearest_photo_vlad(vlad_class, vlad_dataset, args.query)

    elif args.mode == 'create_test_dataset':
        create_test_dataset(dataset.dataset_dir)
    else:
        print('Wrong flags.')
