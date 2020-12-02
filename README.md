# Classification System


## Prerequisite
* Python 3.7.3
* use "pip install -r requirement.txt" in command line to download all required packages


## Input
1. training data: this folder contains two subfolders including the target species and another non-target species (if the extracted features of training data exist, training set is not required)
2. data to be classified: images of interest to be classified using trained classifier
3. parameters in run_classifier.sh file

   One sample input in bash file could be:

   python classifier.py --model_dir imagenet --species human --pred_dir ../pred_images --train_dir ../train --augmentation True --aug_size 1 --sample_size 50 --model MLP

   The parameters:
   
   * model_dir: pre-trained model
   
   * species: species to be classified

   * pred_dir: relative directory of data to be predicted (this can be a directory that contains multiple subfolders)

   * train_dir: relative path of training data

   * augmentation: augment images or not

   * aug_size: augmentation times of a single image in the directory (e.g. if the aug_size=2, each image in the directory would be used for generating 2 more images)

   * sample_size: the number of images to be used for training in each class

   * model: classification methods (GNB/SVM/ET/RF/MLP/KNN/LDA/QDA)
    
      SVM: Support Vector Machine; RF: Random Forest; ET: Extra Trees; MLP: Multi-layer Perceptron; GNB: Naive Bayes; LDA: Linear discriminant analysis; QDA: Quadratic discriminant analysis; KNN:
    k-nearest neighbors

## Running scripts

Compile "run_classifier.sh" in command line

## Saved files
* features of training data (speciesname_features)
* labels of training data (speciesname_labels)
* t-sne features of training data (speciesname_tsne_features.npz)

## Output Results

The output results are saved in Results/ directory.
* t-SNE features
* Confusion Matrix
* csv file contains filenames, probabilities and predicted classifications
