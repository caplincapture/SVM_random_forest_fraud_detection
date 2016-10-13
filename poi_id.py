#!/usr/bin/python

### Part 0 Setup and import library 
import numpy as np
import time
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary' , 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
                 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

print "number of starting features: ",len(features_list) - 1

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

# looking at summary statistics, information is limited and may to look at each entry (i.e. by person)
from scipy.stats import describe
for feat in features_list:
    feat_summary=[]
    for name in data_dict:
        if data_dict[name][feat] != "NaN":   
            feat_summary.append(data_dict[name][feat])
    #print 'feature', feat
    #print 'stats', describe(feat_summary)

# easier iterating for dictionaries within dictionaries    
def lookup(dic, key, *keys):
    if keys:
        return lookup(dic.get(key, {}), *keys)
    return dic.get(key)

# list of people 
names=[]
def list_people(dic, emptylist):
    for name, value in dic.items():
        emptylist.append(name)
    return emptylist
        
list_people(data_dict, names)

# outputs entries w/ only values listed, 'TOTAL'(sum of all the values), 'LOCKHART EUGENE E' (all NaN values) are removed
# 'THE TRAVEL AGENCY IN THE PARK' is also removed
def person_values(dic):
    for name in names:
        return 'person', name , lookup(dic, name).values()
        #print 'person', name , lookup(dic, name).values()
person_values(data_dict)
    
data_dict.pop('TOTAL',0)
data_dict.pop('LOCKHART EUGENE E',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

print "number of people in the dataset: ", len(data_dict)

# function to print the implicated names in the dataset
# print "print out the person names in the dataset: "
s = []
for person in data_dict.keys():
    s.append(person)
    if len(s) == 4:
      #  print '{:<30}{:<30}{:<30}{:<30}'.format(s[0],s[1],s[2],s[3])
        s = []

# printing out the number of people of interest
npoi = 0
for p in data_dict.values():
    if p['poi']:
        npoi += 1
print "number of people of interest is: ", npoi
print "number of people who are not of interset is: ", len(data_dict) - npoi

# printing out the number of missing values in each feature
print "the number of missing values in each feature: "
NaNInFeatures = [0 for i in range(len(features_list))]
for i, person in enumerate(data_dict.values()):
    for j, feature in enumerate(features_list):
        if person[feature] == 'NaN':
            NaNInFeatures[j] += 1

for i, feature in enumerate(features_list):
    print feature, NaNInFeatures[i]

# sum of all the salaries for people in the data set
salary  = []
for name, person in data_dict.iteritems():
    if float(person['salary']) > 0:
        salary.append(float(person['salary']))
print "the salary sum of all the remaining people in the data set: ",np.sum(salary) 
    

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# creating a variable denoting the proportion of messages to and from people of interest
print "creating two new features representing the proportion of inbound and outbound messages that were received from and sent to people of interest"

for person in my_dataset.values():
    person['to_poi_message_ratio'] = 0
    person['from_poi_message_ratio'] = 0
    if float(person['from_messages']) > 0:
        person['to_poi_message_ratio'] = float(person['from_this_person_to_poi'])/float(person['from_messages'])
    if float(person['to_messages']) > 0:
        person['from_poi_message_ratio'] = float(person['from_poi_to_this_person'])/float(person['to_messages'])
    
features_list.extend(['to_poi_message_ratio', 'from_poi_message_ratio'])

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


'''
    
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
'''
# next steps-tune the train, test split and stratified shuffle split to make a set training and testing fit
folds=1000
sss=StratifiedShuffleSplit(labels, folds, random_state=42)
import timeit

# Gaussian naive bayes model, feature scaled, select k best, PCA all used

start = timeit.default_timer()
steps_gnb = [('feature_scaling', StandardScaler()),
             ('skb', SelectKBest()),
                ('reduce_dim', PCA()), 
                ('GNB', GaussianNB())]


GNB = Pipeline(steps_gnb)
params_gnb = dict( reduce_dim__n_components=[5, 6, 7, 8, 9, 10])
clf_gnb = GridSearchCV(GNB, param_grid=params_gnb, cv=sss, scoring='f1')
clf_gnb.fit(features, labels)
best_gnb=clf_gnb.best_estimator_
best_gnb_features=best_gnb.named_steps['skb'].get_support()
indices = np.argsort(best_gnb_features)[::-1]
print 'Feature Ranking: '
for i in range(3):
    print "feature no. {}: {} ({})".format(i+1,features_list[indices[i]+1],best_gnb_features[indices[i]])

print "The naive bayes' selected features are:\n", [x for x, y in zip(features_list[1:], best_gnb_features) if y]
print "Explained variance along x number of axes: \n", best_gnb.named_steps['reduce_dim'].explained_variance_
print 'GNB report:', test_classifier(best_gnb, my_dataset, features_list, folds=1000)
stop = timeit.default_timer()
print stop - start

# decision tree with select k best

start1 = timeit.default_timer()
steps_dtc=[('scaling', MinMaxScaler()), ("DTC", DecisionTreeClassifier())]
DTC=Pipeline(steps_dtc)
parameters = {}
clf_dtc = GridSearchCV(DTC, parameters, scoring='f1', cv = sss)
clf_dtc.fit(features, labels)
best_dtc=clf_dtc.best_estimator_

# feature rankings

print 'dtc report:', test_classifier(best_dtc, my_dataset, features_list, folds=1000)

stop1 = timeit.default_timer()

print stop1 - start1


# Random Forest with select kbest

start2 = timeit.default_timer()
steps_rf = [('skb', SelectKBest()),
                 ('rf', RandomForestClassifier(n_estimators=100,))]

rf = Pipeline(steps_rf)
params_rf = dict(skb=range(4,10))
clf_rf = GridSearchCV(rf, param_grid=params_rf, scoring='f1', cv=sss)
clf_rf.fit(features, labels)
best_rf=clf_rf.best_estimator_
# best random forest features are printed below
rf_selected=best_rf.named_steps['skb'].get_support()
print "The random forest' selected features are:\n", [x for x, y in zip(features_list[1:], rf_selected) if y]

importances_rf=best_rf.named_steps['rf'].feature_importances_
print 'feature importances', importances_rf
'''
'''
indices2 = np.argsort(importances_rf)[::-1]
print 'Feature Ranking: '
for i in range(3):
    print "feature no. {}: {} ({})".format(i+1,features_list[indices2[i]+1],importances_rf[indices2[i]])
'''
'''    
print 'rf report:', test_classifier(best_rf, my_dataset, features_list, folds=1000)

stop2 = timeit.default_timer()

print stop2 - start2

# SVC, ~12 hours of training time with feature scaler and select k best

start3 = timeit.default_timer()

steps_svc = [('skb', SelectKBest()),
                ('feature_scaling', MinMaxScaler()), 
                 ('svm', SVC(kernel='linear', cache_size=2500))]

svc = Pipeline(steps_svc)
C_range = np.logspace(2, 5, 4)
gamma_range = np.logspace(-3, 0, 4)
params_svc = dict(svm__gamma=gamma_range, svm__C=C_range)
clf_svc = GridSearchCV(svc, param_grid=params_svc, cv=sss, scoring='f1')
clf_svc.fit(features, labels)
best_svc=clf_svc.best_estimator_
# feature's importances and select k best best features
importances_svc=best_svc.named_steps['svm'].support_
svc_best_features=best_svc.named_steps['skb'].get_support()
print "The support vector classifier's selected features are:\n", [x for x, y in zip(features_list[1:], svc_best_features) if y]


print test_classifier(best_svc, my_dataset, features_list, folds=1000)
stop3 = timeit.default_timer()

print stop3 - start3


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = best_gnb
dump_classifier_and_data(clf, my_dataset, features_list)
