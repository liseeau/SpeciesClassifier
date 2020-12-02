import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
from numpy import expand_dims
import matplotlib.pyplot as plt
import pickle
import collections
import itertools
import time
import os
import re
import sys
from PIL import Image
import pandas as pd
import seaborn as sn
import argparse
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import random


# parameters to be passed through .sh file or command line
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, help = "directory of pre-trained model", default = 'imagenet')
parser.add_argument("--species", type=str, help = "Species to be classified", required=True)
parser.add_argument("--pred_dir", type=str, help = "The relative path of images for prediction", required = True)
parser.add_argument("--train_dir", type=str, help = "The relative path of images for training", required = False, default=None)
parser.add_argument("--augmentation", type=bool, help = "Augment images or not", default=False)
parser.add_argument("--aug_size", type=int, help = "augmentation times of a single image in the directory")
parser.add_argument("--sample_size", type=int, help = "sampling size", default=0)
parser.add_argument("--model", type=str, help = "choose classification model from GNB/SVM/ET/RF/MLP/KNN/LDA/QDA", default = 'MLP')
args = parser.parse_args()


def TrainProcessing(path, save_dir):
    """
    Image cropping and compression for training set
    """
    print("Processing training images...")
    # create a temp directory to store images
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for file in os.listdir(path):
        if re.search('jpg|JPG', file):
            picture = Image.open(f'{path}/{file}')
            w, h = picture.size
            # crop the info-bar from image
            img = picture.crop((0, 30, w, h-100))
            # compress image if the size is greater than 1MB
            if os.stat(f'{path}/{file}').st_size >1000000:
                img.save(f'{save_dir}/{file}', "JPEG", optimize=True, quality=85)
            else:
                img.save(f'{save_dir}/{file}')

def PredProcessing(path, save_dir):
    """
    Image cropping and compression for images to be classified
    """
    print("Processing images to be classified...")

    list_pred_images=[]
    for root, dirs, files in os.walk(path):
        list_pred_images.extend([os.path.abspath(os.path.join(root,file)) for file in files if re.search('jpg|JPG', file)])

    i = 0
    remove_list = []
    for image in list_pred_images:
        try:
            picture = Image.open(image)
            w, h = picture.size
            img = picture.crop((0, 30, w, h-100))
            if os.stat(image).st_size >1000000:
                img.save(f'{save_dir}/{i}.JPG', "JPEG", optimize=True, quality=85)
            else:
                img.save(f'{save_dir}/{i}.JPG')
            i += 1
        except:
            remove_list.append(image)
    list_pred_images = [img for img in list_pred_images if img not in remove_list]
    return list_pred_images

def augmentation(path, aug_size, dir_name):
    """Image augmentation"""
    print("augmenting images...")
    # load the image
    img_dir = path
    for file in os.listdir(img_dir):
        img = load_img(f'{img_dir}/{file}')
        # convert to numpy array
        image = img_to_array(img)
        # expand dimension to one sample
        sample = expand_dims(image, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(horizontal_flip=True,zoom_range=[0.6,1.0],width_shift_range=[-100,100],brightness_range=[0.5,1.0])
        # save directory
        save_dir = dir_name
        datagen.fit(sample)
        for x, val in zip(datagen.flow(sample,                    #image we chose
            save_to_dir=save_dir,     #this is where we figure out where to save
            save_prefix=file[:-5],      # it will save the images as 'aug_0912' some number for every new augmented image
            save_format='jpg'),range(aug_size)) :     # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
            pass
    
def sampling(path, sample_size):
    """Sampling to get balanced training data"""
    image_list = [path + '/' + f for f in os.listdir(path) if re.search('jpg|JPG', f)]
    sample_list = random.sample(image_list, sample_size)
    return sample_list

def create_graph():
    """Create the CNN graph"""
    with gfile.FastGFile(os.path.join(args.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
#        graph_def = tf.GraphDef()  # this is for python 3.6
        graph_def = tf.compat.v1.GraphDef()  # for python 3.7/tensorflow 2.0
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(list_images):
    """Extract bottleneck features"""
    nb_features = 2048
    features = np.empty((len(list_images), nb_features))
    labels = []

    create_graph()

    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048 float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG encoding of the image.

    #with tf.Session() as sess:  # this is for python 3.6
    with tf.compat.v1.Session() as sess:  # for python 3.7/tensorflow 2.0
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(list_images):
            imlabel = image.split('/')[-2]

            # rough indication of progress
            if ind % 100 == 0:
                print('Processing', image, imlabel)
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[ind, :] = np.squeeze(predictions)
            labels.append(imlabel)

    return features, labels


def plot_features(feature_labels, t_sne_features):
    """feature plot"""
    plt.figure(figsize=(9, 9), dpi=100)
    uniques = {x: feature_labels.count(x) for x in feature_labels}
    od = collections.OrderedDict(sorted(uniques.items()))

    colors = itertools.cycle(["r", "b"])
    n = 0
    for label in od:
        count = od[label]
        m = n + count
        plt.scatter(t_sne_features[n:m, 0], t_sne_features[n:m, 1], c=next(colors), s=10, edgecolors='none')
        c = (m + n) // 2
        plt.annotate(label, (t_sne_features[c, 0], t_sne_features[c, 1]))
        n = m
    plt.savefig("Results/t_SNE_features")


def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10), Actual = "Actual", Predicted = "Predicted"):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = Actual
    cm.columns.name = Predicted
    fig, ax = plt.subplots(figsize=figsize)
    sn.heatmap(cm, annot=annot, fmt='', ax=ax, cmap='Blues')
    plt.savefig("Results/Confusion_Matrix.png")

def run_classifier(model, pred_features, x_train_data, y_train_data, x_test_data, y_test_data, acc_str, matrix_header_str):
    """run chosen classifier and display results"""
    if model == "GNB":
        clfr = GaussianNB()
    elif model == "SVM":
        clfr = LinearSVC()
    elif model == "ET":
        clfr = ExtraTreesClassifier(n_jobs=4,  n_estimators=100, criterion='gini', min_samples_split=10,
                          max_features=50, max_depth=40, min_samples_leaf=4)
    elif model == "RF":
        clfr = RandomForestClassifier(n_jobs=4, criterion='entropy', n_estimators=70, min_samples_split=5)
    elif model == "KNN":
        clfr = KNeighborsClassifier(n_neighbors=1, n_jobs=4)
    elif model == "MLP":
        clfr = MLPClassifier()
    elif model =="LDA":
        clfr = LinearDiscriminantAnalysis()
    elif model =="QDA":
        clfr = QuadraticDiscriminantAnalysis()

    clfr.fit(x_train_data, y_train_data)
    y_pred = clfr.predict(x_test_data)

    # confusion matrix computation and display
    accuracy = acc_str.format(accuracy_score(y_test_data, y_pred) * 100)
    print(accuracy)
    cm_analysis(y_test_data, y_pred, list(set(y_test_data)))

    # predictions
    pred_probs = clfr.predict_proba(pred_features)
    preds = clfr.predict(pred_features)
    return pred_probs, preds

def get_features(species, list_images, pred_dir):
    """get and save extracted features and labels in pickle files"""
    # get images - labels are from the subdirectory names
    if os.path.exists(species.lower()+'_features'):
        print('Pre-extracted features and labels found. Loading them ...')
        features = pickle.load(open(species.lower()+'_features', 'rb'))
        labels = pickle.load(open(species.lower()+'_labels', 'rb'))
    else:
        print('No pre-extracted features - extracting features ...')

        # extract features
        features, labels = extract_features(list_images)

        # save, so they can be used without re-running the last step which can be quite long
        pickle.dump(features, open(species.lower()+'_features', 'wb'))
        pickle.dump(labels, open(species.lower()+'_labels', 'wb'))
        print('CNN features obtained and saved.')

    ### extract features for prediction
    print('extracting features for prediction...')
    index = sorted([int(f.split('.')[0]) for f in os.listdir(pred_dir) if re.search('jpg|JPG', f)])
    list_pred_images = [f'{pred_dir}{idx}.JPG' for idx in index]
    pred_features, _ = extract_features(list_pred_images)
    return features, labels, pred_features

def plot_tsne_features(species, features, labels):
    """plot t-sne features"""
    # t-sne feature plot
    if os.path.exists(species+'_tsne_features.npz'):
        print('t-sne features found. Loading ...')
        tsne_features = np.load(species+'_tsne_features.npz')['tsne_features']
    else:
        print('No t-sne features found. Obtaining ...')
        tsne_features = TSNE().fit_transform(features)
        np.savez(species+'_tsne_features', tsne_features=tsne_features)
        print('t-sne features obtained and saved.')

    plot_features(labels, tsne_features)

def main():
    # record start time
    start_time = time.time()
    # check whether the extracted features have already exist
    if not os.path.exists((args.species).lower()+'_features'):
        # make a temporary directory 'train_processed' and 'pred_processed' to store processed data
        if not os.path.exists('../train_processed'):
            os.mkdir('../train_processed')
        if not os.path.exists('../pred_processed'):
            os.mkdir('../pred_processed')
        dir_list = [x[0] for x in os.walk(args.train_dir)]
        dir_list = dir_list[1:]
        sample_list = []
        for i in dir_list:
            # process training images
            TrainProcessing(i, '../train_processed/' + i.split('/')[-1])
            # image augmentation
            if args.augmentation:
                augmentation('../train_processed/' + i.split('/')[-1], args.aug_size-1, '../train_processed/' + i.split('/')[-1])
            # sampling
            if args.sample_size:
                temp = sampling('../train_processed/'+ i.split('/')[-1], args.sample_size)
                sample_list.extend(temp)
            else:
                temp = ['../train_processed/'+ i.split('/')[-1] + '/'+ f for f in os.listdir('../train_processed/'+ i.split('/')[-1]) if re.search('jpg|JPG', f)]
                sample_list.extend(temp)
        #process data for prediction and get features
        pred_images_list = PredProcessing(args.pred_dir, '../pred_processed') #!!!
        features, labels, pred_features = get_features(args.species, sample_list, "../pred_processed/")
    
    else:
        if not os.path.exists('../pred_processed'):
            os.mkdir('../pred_processed')
        pred_images_list = PredProcessing(args.pred_dir, '../pred_processed')
        features, labels, pred_features = get_features(args.species, [], "../pred_processed/")
   
    plot_tsne_features(args.species, features, labels)
    
    # prepare training and test datasets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=20)

    pred_probs, preds = run_classifier(args.model, pred_features, X_train, y_train, X_test, y_test, "CNN-MLP Accuracy: {0:0.1f}%",
               "Multi-layer Perceptron Confusion matrix")

    # print runtime for training classifier and tagging images to be classified
    print("runtime: %s seconds" % (time.time() - start_time))

    df = pd.DataFrame()
    df["Filename"] = pred_images_list
    df["probability"] = [j[0] for j in pred_probs]
    df[args.species] = preds
    df.to_csv("Results/predictions.csv", index = False)

if __name__ == "__main__":
	main()