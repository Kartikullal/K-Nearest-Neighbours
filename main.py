
from KNN import KNN
from parse_images import parse_images 
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    print('Pulling Training Data')
    train_path = 'Data/train-images-idx3-ubyte'
    train_label_path = 'Data/train-labels-idx1-ubyte'
    
    train_parser = parse_images(train_path, train_label_path)
    X_train = train_parser.parse_images()
    y_train = train_parser.parse_labels()
    print('Train data Acquired')
    print()
    print('Pulling Test Data')
    test_path = 'Data/t10k-images-idx3-ubyte'
    test_label_path = 'Data/t10k-labels-idx1-ubyte'

    test_parser = parse_images(test_path, test_label_path)
    X_test= test_parser.parse_images()
    y_test = test_parser.parse_labels()
    print('Test Data Acquired')
    print()
    #Fit the data to the model

    print('Training Model')
    model = KNN(k = 10, metric = 'eucledian')
    model.fit(X_train, y_train)
    print('Model Trained')
    print()

    print('Testing Model')
    count = 0
    i = 0
    y_pred = []
    y_pred = model.predict(X_test)
    print('Printing Accuracy')
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy Score: ', accuracy)