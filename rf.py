# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 11:58:29 2018

@author: A89X2ZZ
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold # import KFold
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
# Load numpy
import numpy as np

np.random.seed(0)
"""
X_train = pd.read_csv('/Users/A89X2ZZ/Documents/0 - Personal/Machine Learning/project/data version2/data version2/X_train.csv')
Y_train = pd.read_csv('/Users/A89X2ZZ/Documents/0 - Personal/Machine Learning/project/data version2/data version2/Y_train.csv')
X_test= pd.read_csv('/Users/A89X2ZZ/Documents/0 - Personal/Machine Learning/project/data version2/data version2/X_test.csv')
Y_test = pd.read_csv('/Users/A89X2ZZ/Documents/0 - Personal/Machine Learning/project/data version2/data version2/Y_test.csv')
#ChinPred = pd.read_csv('/Users/A89X2ZZ/Documents/0 - Personal/Machine Learning/project/data version2/data version2/pred_l.csv')
"""

path ="/Users/A89X2ZZ/Documents/0 - Personal/Machine Learning/project/data version2/data version2/"
X_train=pd.read_csv(path+"X_train.csv")
Y_train=pd.read_csv(path+"Y_train.csv",header=None)
Y_train = Y_train.iloc[1:]
X_test=pd.read_csv(path+"X_test.csv")
Y_test=pd.read_csv(path+"Y_test.csv",header=None)
Y_test = Y_test.iloc[1:] 
X_train[0:5]

#clf = RandomForestClassifier(n_jobs=2, random_state=0,n_estimators= 3)
#clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None,min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False,class_weight=None)
clf = RandomForestClassifier(n_estimator = 25, oob_score = TRUE, n_jobs = -1,random_state =50,  max_features = "auto", min_samples_leaf = 50)
clf.fit(X_train, Y_train)

preds = clf.predict(X_test)
#preds = clf.predict_proba(X_test)[0:10]
preds[0:5]
Y_test[0:5]
print(X_train.feature_names)

df = pd.DataFrame(X_train, columns=X_train.feature_names)

import pandas as pd
feature_imp = pd.Series(clf.feature_importances_,index=X_train.feature_names).sort_values(ascending=False)
feature_imp



def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    
    # Plot the training points
    ax.scatter(X[:, 1], X[:, ], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
    
    
    
visualize_classifier(clf, X_train,Y_train);



pd.crosstab(Y_test['Season'], preds, rownames=['Actual'], colnames=['Predicted '])
    accuracy_score(Y_train, clf.predict(X_train))
   accuracy_score(Y_test, preds)
   accuracy_score(Y_test, ChinPred)
confusion_matrix(Y_test, preds)
print(metrics.classification_report(Y_test,preds))

from sklearn import tree
i_tree = 0
for tree_in_forest in clf.estimators(3):
    export_graphviz(tree_in_forest,
                feature_names=X_train.columns,
                filled=True,
                rounded=True)
    os.system('dot -Tpng tree.dot -o tree.png')
    
import pydotplus
import six
from sklearn import tree
dotfile = six.StringIO()    
from sklearn.tree import export_graphviz
for tree_in_forest in clf.estimators_:
    if (i_tree <1):        
        export_graphviz(tree_in_forest,
                feature_names=X_train.columns,
                filled=True,
                rounded=True)
os.system('dot -Tpng tree.dot -o tree.png')
dot foo.dot -Tpng -o foo.png
dot -Tpng tree.dot -o tree.png
draw_tree(clf)

def draw_tree(ensemble, tree_id=0):

     plt.figure(figsize=(8,8))
     plt.subplot(211)

     tree = ensemble.estimators_[tree_id].tree_

     depths = leaf_depths(tree)
     plt.hist(depths, histtype='step', color='#9933ff', 
              bins=range(min(depths), max(depths)+1))

     plt.xlabel("Depth of leaf nodes (tree %s)" % tree_id)
    
     plt.subplot(212)
    
     samples = leaf_samples(tree)
     plt.hist(samples, histtype='step', color='#3399ff', 
              bins=range(min(samples), max(samples)+1))
    
     plt.xlabel("Number of samples in leaf nodes (tree %s)" % tree_id)
   
     plt.show()
     
     
     
     
tree.export_graphviz(clf,out_file='tre1e.dot') 
'''
from sklearn.tree import export_graphviz
clf = RandomForestClassifier(n_estimators=10)
import pydotplus
import six
from sklearn import tree
dotfile = six.StringIO()
i_tree = 0
for tree_in_forest in clf.estimators_:
 if (i_tree <1):        
    tree.export_graphviz(tree_in_forest, out_file=dotfile)
    pydotplus.graph_from_dot_data(dotfile.getvalue()).write_png('dtree'+ str(i_tree) +'.png')
    i_tree = i_tree + 1
    '''
    
    
#compute roc
from sklearn.metrics import roc_curve,roc_auc_score,auc

pred_0 = clf.predict_proba(X_test)[:,0]
pred_1 = clf.predict_proba(X_test)[:,1]


# Compute ROC curve and ROC area for each class
n_classes = 2
fpr_0, tpr_0, threshold_0 = roc_curve(Y_test, pred_0,0)
roc_auc_0 = auc(fpr_0, tpr_0)
fpr_1, tpr_1, threshold_1 = roc_curve(Y_test, pred_1,1)
roc_auc_1 = auc(fpr_1, tpr_1)

plt.figure()
lw = 2
fig, ax = plt.subplots(figsize=(18,8))

plt.plot(fpr_0, tpr_0, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_0)
plt.plot(fpr_1, tpr_1, color='green',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_1)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()



'''

from sklearn.metrics import roc_curve,roc_auc_score,auc

pred_0 = clf.predict_proba(X_test)[:,0]#random forest
pred_1 = clf.predict_proba(X_test)[:,0]#logit
pred_2 = clfd.predict_proba(X_test)[:,0]#svm


# Compute ROC curve and ROC area for each class
n_classes = 2
fpr_0, tpr_0, threshold_0 = roc_curve(Y_test, pred_0,0)
roc_auc_0 = auc(fpr_0, tpr_0)
fpr_1, tpr_1, threshold_1 = roc_curve(Y_test, pred_1,0)
roc_auc_1 = auc(fpr_1, tpr_1)
fpr_2, tpr_2, threshold_1 = roc_curve(Y_test, pred_2,0)
roc_auc_2 = auc(fpr_2, tpr_2)

plt.figure()
lw = 2
fig, ax = plt.subplots(figsize=(18,8))

plt.plot(fpr_0, tpr_0, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_0)
plt.plot(fpr_1, tpr_1, color='green',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_1)
plt.plot(fpr_2, tpr_2, color='green',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_2)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()






"""
#class 1 roc
 y_pred_proba1 = clf.predict_proba(X_test)[::,1]
    auc = metrics.roc_auc_score(Y_test, y_pred_proba1)
    fpr2, tpr2, thresholds = metrics.roc_curve(Y_test,y_pred_proba1,)

    y_pred_proba = clf.predict_proba(X_test)[::,0]
    auc = metrics.roc_auc_score(Y_test, y_pred_proba)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test,y_pred_proba, pos_label=0)
    
    plt.figure()
    lw = 2
    plt.plot(fpr,tpr,label="data1")
    plt.plot(fpr2,tpr2)
    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for 0')
    plt.legend(loc="lower right")
    plt.show()

 print(\"Accuracy\", metrics.accuracy_score(Y_test, pred))

import scikitplot as skplt
import matplotlib.pyplot as plt

y_true = # ground truth labels
y_probas = # predicted probabilities generated by sklearn classifier
skplt.metrics.plot_roc_curve(Y_test preds[:,1])
plt.show()


















preds.to_csv('result.csv', index=False, sep=',')
# f1, precision, 
path='/Users/A89X2ZZ/Documents/0 - Personal/Machine Learning/project/ '
preds.to_csv(path,'greenl.csv')

print(classification_report(estY,preds))

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb

print("Train a XGBoost model")
params = {"objective": "reg:linear",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=250
gbm = xgb.train(params, xgb.DMatrix(
    X_train, Y_train), num_trees)

vc =gbm.predict(xgb.DMatrix(X_test))
from decimal import *
vc1 = [round(value) for value in vc]
vc2 = int(round(float(vc1)))
print(type(vc))


vc3 = int(vc.split('.')[0])
vc1[0:5]
describe(vc1)
import numpy as np

print("Convert float values to integer values:")

vc = (vc.astype(int))


import numpy as np
x= np.array([[12.0, 12.51], [2.34, 7.98], [25.23, 36.50]])
print("Original array elements:")
print(x)
print("Convert float values to integer values:")
print(x.astype(int))
accuracy_score(Y_test, vc)

print("Make predictions on the test set")
test_probs = (clf.predict_proba(X_test) +
              gbm.predict(xgb.DMatrix(X_test)))/2
    c1 = clf.predict_proba(X_test)  
 x1= gbm.predict(xgb.DMatrix(X_test))        
 i1 = x1+c1/2             

              
  from xgboost import XGBClassifier            
model = xgb.XGBClassifier()
model.fit(X_train, Y_train)
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in testpro]
testpro = (y_pred+ preds)/2
accuracy = accuracy_score(Y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

result = pd.DataFrame({'id': X_test.index})
result['Season'] = clf.predict_proba(X_test)[:, 1]
result.to_csv('result.csv', index=False, sep=',')

 accuracy_score(Y_test, preds)




#To plot

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=2)
pca.fit(X_train)


X_train = pca.transform(X_train)

for name, label in [( 0), ( 1)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
             
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    
    # Reorder the labels to have colors matching the cluster results
Y_train = np.choose(Y_train, [0,1]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y_train, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()


X_pca = decomposition.PCA(n_components=2).fit_transform(X_train)
embedding_plot(X_pca, "PCA")
plt.show()"""