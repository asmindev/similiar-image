import os
import cv2
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from keras.applications.xception import Xception, preprocess_input
from matplotlib import pyplot as plt
from joblib import dump, load


class ImageSearch:
    def __init__(
        self,
        images_dir="static/img/dataset-image",
        model_file="xception_model.pkl",
        feature_file="feature_vec.joblib",
    ):
        self.images_dir = images_dir
        self.model_file = f"{self.get_current_dir()}/{model_file}"
        self.feature_file = f"{self.get_current_dir()}/{feature_file}"
        self.main_model = self.load_model()
        self.feature_vec = self.load_features()

    def get_current_dir(self):
        return os.path.dirname(os.path.realpath(__file__))

    def getImagePaths(self):
        image_names = []
        for dirname, _, filenames in os.walk(self.images_dir):
            for filename in filenames:
                fullpath = os.path.join(dirname, filename)
                image_names.append(fullpath)
        return image_names

    def preprocess_img(self, img_path):
        dsize = (225, 225)
        new_image = cv2.imread(img_path)
        new_image = cv2.resize(new_image, dsize, interpolation=cv2.INTER_NEAREST)
        new_image = np.expand_dims(new_image, axis=0)
        new_image = preprocess_input(new_image)
        return new_image

    def load_data(self):
        output = self.getImagePaths()
        return output

    def create_model(self):
        model = Xception(weights="imagenet", include_top=False)
        for layer in model.layers:
            layer.trainable = False
        return model

    def load_model(self):
        print("Loading model...")
        if os.path.exists(self.model_file):
            print(f"Loading existing model from: {self.model_file}")
            with open(self.model_file, "rb") as f:
                model = pickle.load(f)
        else:
            print(f"Creating new model: {self.model_file}")
            model = self.create_model()
            with open(self.model_file, "wb") as f:
                pickle.dump(model, f)
        return model

    def extract_features(self):
        features = []
        output = self.load_data()
        for i in output[:999]:
            new_img = self.preprocess_img(i)
            new_feature = self.main_model.predict(new_img)
            new_feature = np.array(new_feature)
            new_feature = new_feature.flatten()
            features.append(new_feature)
        feature_vec = np.array(features)
        return feature_vec

    def save_features(self, feature_vec):
        dump(feature_vec, self.feature_file)

    def load_features(self):
        if os.path.exists(self.feature_file):
            print(f"Loading features from {self.feature_file}")
            return load(self.feature_file)
        else:
            print("Extracting features...")
            feature_vec = self.extract_features()
            self.save_features(feature_vec)
            return feature_vec

    def search_similar_images(self, query_img_path, n_results=12):
        query_img = self.preprocess_img(query_img_path)
        query_feature = self.main_model.predict(query_img)
        query_feature = np.array(query_feature)
        query_feature = query_feature.flatten()
        nbrs = NearestNeighbors(n_neighbors=n_results, metric="cosine").fit(
            self.feature_vec
        )
        distances, indices = nbrs.kneighbors([query_feature])
        return indices

    def search_similar_images(self, query_img_path, n_results=12):
        query_img = self.preprocess_img(query_img_path)
        query_feature = self.main_model.predict(query_img)
        query_feature = np.array(query_feature)
        query_feature = query_feature.flatten()
        nbrs = NearestNeighbors(n_neighbors=n_results, metric="cosine").fit(
            self.feature_vec
        )
        distances, indices = nbrs.kneighbors([query_feature])

        output = self.load_data()
        result_paths = [output[index] for index in indices[0]]
        return result_paths

    def search_and_get_results(self, query_img_path):
        result_paths = self.search_similar_images(query_img_path)
        return list(
            map(
                lambda path: {"url": path, "name": path.split("/")[-1].split(".")[0]},
                result_paths,
            )
        )
