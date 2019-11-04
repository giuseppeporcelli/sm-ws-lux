from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn import tree
from sklearn.externals import joblib
from sklearn import metrics

def measure_performance(X,y,clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True):
    y_pred=clf.predict(X)   
    if show_accuracy:
        print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n")

    if show_classification_report:
        print ("Classification report")
        print (metrics.classification_report(y,y_pred),"\n")
        
    if show_confusion_matrix:
        print ("Confusion matrix")
        print (metrics.confusion_matrix(y,y_pred),"\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--max_leaf_nodes', type=int, default=os.environ['SM_HP_MAX_LEAF_NODES'])
    parser.add_argument('--max_depth', type=int, default=os.environ['SM_HP_MAX_DEPTH'])

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val', type=str, default=os.environ['SM_CHANNEL_VAL'])

    args = parser.parse_args()

    # Load training data.
    train_data = pd.read_csv(args.train + '/windturbine_data_train.csv', header=None, engine="python") 
    # labels are in the first column
    train_y = train_data.ix[:,0]
    train_X = train_data.ix[:,1:]
    
    # Load validation data.
    validation_data = pd.read_csv(args.val + '/windturbine_data_val.csv', header=None, engine="python")
    # labels are in the first column
    val_y = validation_data.ix[:,0]
    val_X = validation_data.ix[:,1:]

    # Here we support a single hyperparameter, 'max_leaf_nodes'. Note that you can add as many
    # as your training my require in the ArgumentParser above.
    max_leaf_nodes = args.max_leaf_nodes
    max_depth = args.max_depth

    # Now use scikit-learn's decision tree classifier to train the model.
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, max_depth=max_depth)
    clf = clf.fit(train_X, train_y)
    
    print('Training metrics: \n')
    measure_performance(train_X, train_y, clf, show_classification_report=True, show_confusion_matrix=True)
    
    print('Validation metrics: \n')
    measure_performance(val_X, val_y, clf, show_classification_report=True, show_confusion_matrix=True)
    
    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))

def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf