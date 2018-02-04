#!/usr/bin/env python
"""
authorship.py: This file provides a template to practice how to do authorship attribution (classification) in Python
The steps in this template can be used for any simple classification task. It contains the following steps:
- loading of the data
- splitting into a train, validation and test set.
- Extracting of the features (exercise)
- Classification (i.e. SVM, but easily replace with other classifiers)
- Evaluation (recall, precision, F1)

Good luck and have fun!
"""

__author__      = "Simon Brugman, Christoph Aurnhammer"
__license__     = "MIT"

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
import nltk
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.svm import SVC


# Download the 'stopwords' and 'punkt' from the Natural Language Toolkit,
#  you can comment the next lines if already present.
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


# Load the dataset into memory from the filesystem
def load_data(dir_name):
    return sklearn.datasets.load_files('data/%s' % dir_name, encoding='utf-8')


def load_train_data():
    return load_data('train')


def load_test_data():
    return load_data('test')


# Extract features from a given text
def extract_features(text):
    bag_of_words = [x for x in wordpunct_tokenize(text)]

    features = []
    # Example feature 1: count the number of words
    features.append(len(bag_of_words))

    # Example feature 2: count the number of words, excluded the stopwords
    features.append(len([x for x in bag_of_words if x.lower() not in stop_words]))

    # TODO: Follow the instructions in the assignment and add your own features.

    return features


# Classify using the features
def classify(train_features, train_labels, test_features):
    # TODO: (Optional) If you would like to test different how classifiers would perform different, you can alter
    # TODO: the classifier here.
    clf = SVC(kernel='linear')
    clf.fit(train_features, train_labels)
    return clf.predict(test_features)


# Evaluate predictions (y_pred) given the ground truth (y_true)
def evaluate(y_true, y_pred):
    # TODO: What is being evaluated here and what does it say about the performance? Include or change the evaluation
    # TODO: if necessary.
    recall = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    print("Recall: %f" % recall)

    precision = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    print("Precision: %f" % precision)

    f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    print("F1-score: %f" % f1_score)

    return recall, precision, f1_score


# The main program
def main():
    train_data = load_train_data()

    # Extract the features
    features = list(map(extract_features, train_data.data))

    # Classify and evaluate
    skf = sklearn.model_selection.StratifiedKFold(n_splits=10)
    scores = []
    for fold_id, (train_indexes, validation_indexes) in enumerate(skf.split(train_data.filenames, train_data.target)):
        # Print the fold number
        print("Fold %d" % (fold_id + 1))

        # Collect the data for this train/validation split
        train_features = [features[x] for x in train_indexes]
        train_labels = [train_data.target[x] for x in train_indexes]
        validation_features = [features[x] for x in validation_indexes]
        validation_labels = [train_data.target[x] for x in validation_indexes]

        # Classify and add the scores to be able to average later
        y_pred = classify(train_features, train_labels, validation_features)
        scores.append(evaluate(validation_labels, y_pred))

        # Print a newline
        print("")

    # Print the averaged score
    recall = sum([x[0] for x in scores]) / len(scores)
    print("Averaged total recall", recall)
    precision = sum([x[1] for x in scores]) / len(scores)
    print("Averaged total precision", precision)
    f_score = sum([x[2] for x in scores]) / len(scores)
    print("Averaged total f-score", f_score)
    print("")

    # TODO: Once you are done crafting your features and tuning your model, also test on the test set and report your
    # TODO: findings. How does the score differ from the validation score? And why do you think this is?
    # test_data = load_test_data()
    # test_features = list(map(extract_features, test_data.data))
    #
    # y_pred = classify(features, train_data.target, test_features)
    # evaluate(test_data.target, y_pred)


# This piece of code is common practice in Python, is something like if "this file" is the main file to be ran, then
# execute this remaining piece of code. The advantage of this is that your main loop will not be executed when you
# import certain functions in this file in another file, which is useful in larger projects.
if __name__ == '__main__':
    main()
