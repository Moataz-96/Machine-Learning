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
    f, axes = plt.subplots(math.ceil(num_of_features/2),2, figsize=(35, 25), sharex=True)
    sns.despine(top=True,right=False,left=False,bottom=False)
    ax = 0
    for feature_id in range(num_of_features):
        sns.distplot(data[features[feature_id]],shade=True,color=np.random.rand(3,),ax=axes[(feature_id//2),ax])
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



def read_list_cats(dataset,feature_name,splitter='|',drop_first=True,pca=False,pca_k=2):

    '''
    Params:
            dataset: DataFrame contains all features
            feature_name : specific feature which each row like Comedy|Adv|Horror..
            drop_first: OneHotEncoder without first column
    return:
            Copied of dataset with new feautre and removed original feature
            
    Expected:
            dataset[feature_name] has not any missing values
    '''
    dataset = dataset.copy()
    feature = []
    list(map(lambda x: [feature.append(_) for _ in x.split("|")],dataset[feature_name].values))
    # check if new feature has same name of old feature,append feature_name
    feature_types = list(set(feature))
    feature_types = [feature_name +" " + name for name in feature_types]
    feature_data = np.zeros((len(feature_types),dataset.shape[0]))
    dict_feature = {}
    for type_,data_ in zip(feature_types,feature_data):
        dict_feature[type_] = data_
    for iteration,val in enumerate(dataset[feature_name].values):
        for type_ in val.split(splitter):
            dict_feature[feature_name +" "+ type_][iteration] = 1.0
            
    #let's check our code
    def test_(x,y):
        for i in x.split(splitter):
            assert y[feature_name + " " + i] == 1
    df_feature = pd.DataFrame(dict_feature)        
    test_(dataset[feature_name].values[0],df_feature.iloc[0])
    
    if(drop_first):
        df_feature.drop([df_feature.columns.values[0]], axis=1,inplace=True)
        
    if(pca):
        try:
            from sklearn.decomposition import PCA
            k = pca_k
            pca_dec = PCA(n_components=k)
            column_name = [feature_name + str(feat_id) for feat_id in range(0,k)] 
            df_feature = pd.DataFrame(pca_dec.fit_transform(df_feature),columns=column_name)
        except Exception as e:
            print("PCA Error")
    
    dataset.index = range(0,dataset.shape[0])
    df_feature.index = range(0,df_feature.shape[0])
    dataset =  pd.concat([dataset, df_feature],axis=1)
    return dataset


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
        



