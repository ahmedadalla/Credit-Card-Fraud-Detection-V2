from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def create_model(name:str):
    if name=='lg':
        return LogisticRegression()
    elif name == 'rf':
        return RandomForestClassifier()
    elif name == 'nn':
        return MLPClassifier()
    elif name=="knn":
        return KNeighborsClassifier()
