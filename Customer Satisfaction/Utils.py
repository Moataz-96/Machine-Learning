import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_features(data,features):
    '''
    PARAM features: list of features , string
    '''
    num_of_features = len(features)   
    f, axes = plt.subplots(math.ceil(num_of_features/2),2, figsize=(25, 15), sharex=True)
    sns.despine(top=True,right=False,left=False,bottom=False)
    ax = 0
    for feature_id in range(num_of_features):
        sns.kdeplot(data[features[feature_id]],shade=True,color=np.random.rand(3,),ax=axes[(feature_id//2),ax],bw=0)
        ax = 1 - ax
        

def best_features(data,target,features):
    '''
    data: dataframe , all columns without target
    target: dataframe , target column
    features: list of string features
    return sorted dict features
    '''
    rf_clf = RandomForestClassifier()
    rf_clf.fit(data,target)
    feature_importance = rf_clf.feature_importances_
    imp_features = {}
    
    for i,feature_name in enumerate(features):
        imp_features[feature_name] = feature_importance[i]
        
    #sort features based on values
    imp_features = {k: v for k, v in sorted(imp_features.items(), key=lambda item: item[1],reverse=True)}

    return imp_features




class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
     def __init__(self,K, strategy="manually"): 
        # no *args or **kargs
        '''
        Expected data numpy array
        Params
            strategy: 
                manually -> mix columns manually
                poly_interaction -> mix columns using poly_interaction
                poly_degree -> mix columns based on number of columns
                feature_selector -> no mixing but feature selection

            k : number of features as hyperparameters for gridsearch to be used in
                poly_degree or feature_selector

        Return
           new 2D numpy array
        '''
        self.strategy = strategy
        self.K = K
        
     def fit(self, X, y=None):
        return self # nothing else to do
    
     def transform(self, X, y=None):
        if(self.strategy=="manually"):
            #combine data based on expierience
            return X
        elif(self.strategy=="poly_interaction"):
            poly = PolynomialFeatures(interaction_only=True)
            return poly.fit_transform(X)
        elif(self.strategy=="poly_degree"):
            poly = PolynomialFeatures(2)
            return poly.fit_transform(X)
        elif(self.strategy=="feature_selector"):
            return X[:,:self.K].copy()
        
