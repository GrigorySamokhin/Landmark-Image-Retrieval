import pandas as pd
import os

METADATA_DIR = 'metadata'

class DataSet(object):

    def __init__(self, database_dir):
        '''
        :param database_dir: Directory of dataset

        Create a .csv file and load in DataFrame
        .dataset_dir - Directory of dataset
        .data - DataFrame with paths and classes
        .labels - set with classes
        '''
        self.dataset_dir = database_dir
        self.file_name_csv = METADATA_DIR + os.path.sep + database_dir+'.csv'
        self.create_csv()
        self.data = pd.read_csv(self.file_name_csv)
        self.labels = set(self.data["class_image"])
        os.remove(self.file_name_csv)

    def create_csv(self):
        '''
        Creating a .csv file to load in pandas DataFrame
        '''
        if not os.path.exists(METADATA_DIR):
            os.mkdir(METADATA_DIR)
        if os.path.exists(self.file_name_csv):
            return
        with open(self.file_name_csv, 'w', encoding='UTF-8') as file_temp:
            file_temp.write("image_path,class_image\n")
            for root, _, files in os.walk(self.dataset_dir, topdown=False):
                class_image = root.split(os.path.sep)[-1]
                for name in files:
                    if not name.endswith('.jpg'):
                        continue
                    image_path = os.path.join(root, name)
                    file_temp.write("{},{}\n".format(image_path, class_image))

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels

    def get_data(self):
        return self.data
