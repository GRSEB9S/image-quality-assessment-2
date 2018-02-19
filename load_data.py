import glob
import pickle

def read_data(path):
    X = []
    y = []

    for image in glob.glob(path):
        X.append(image)
        y.append(float(image.split('_')[2].split('.jpg')[0]))
        
    return X, y

