import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def train_and_save():
    iris=load_iris()
    X,y = iris.data, iris.target
    model=RandomForestClassifier()
    model.fit(X,y)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    # Also store the training data distribution for drift baseline
    return y
if __name__=="__main__":
    train_and_save()