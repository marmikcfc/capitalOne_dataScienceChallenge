import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score,
                             average_precision_score,
                             auc,
                             classification_report,
                             confusion_matrix,
                             explained_variance_score,
                             f1_score,
                             fbeta_score,
                             hamming_loss,
                             hinge_loss,
                             jaccard_similarity_score,
                             log_loss,
                             matthews_corrcoef,
                             mean_squared_error,
                             mean_absolute_error,
                             precision_recall_curve,
                             precision_recall_fscore_support,
                             precision_score,
                             recall_score,
                             r2_score,
                             roc_auc_score,
                             roc_curve,
                             zero_one_loss)




# convert to CSV

txt_file_train = r"codetest_train.txt"
csv_file_train = r"codetest_train.csv"
txt_file_test = r"codetest_test.txt"
csv_file_test = r"codetest_test.csv"

# use 'with' if the program isn't going to immediately terminate
# so you don't leave files open
# the 'b' is necessary on Windows
# it prevents \x1a, Ctrl-z, from ending the stream prematurely
# and also stops Python converting to / from different line terminators
# On other platforms, it has no effect


in_txt = csv.reader(open(txt_file_test, "r"), delimiter = '\t')
out_csv = csv.writer(open(csv_file_test, 'w',newline=''))
out_csv.writerows(in_txt)

in_txt = csv.reader(open(txt_file_train, "r"), delimiter = '\t')
out_csv = csv.writer(open(csv_file_train, 'w',newline=''))
out_csv.writerows(in_txt)


#Now let the magic begin



train=pd.read_csv('codetest_train.csv')
test=pd.read_csv('codetest_test.csv')
train['Type']='Train' #Create a flag for Train and Test Data set
test['Type']='Test'
fullData = pd.concat([train,test],axis=0) #Combined both Train and Test Data set

print(fullData.describe())

#categorical variables: f_61 f_121 f_215 f_237
target_col = ["target"]

cat_cols = ['f_61','f_121','f_215','f_237']

num_cols= list(set(list(fullData.columns))-set(cat_cols)-set(target_col))
other_col=['Type'] #Test and Train Data set identifier


num_cat_cols = num_cols+cat_cols # Combined numerical and Categorical variables

#Create a new variable for each variable having missing value with VariableName_NA 
# and flag missing value with 1 and other with 0

for var in num_cat_cols:
    if fullData[var].isnull().any()==True:
        fullData[var]=fullData[var].isnull()*1 


#Impute numerical missing values with mean
fullData[num_cols] = fullData[num_cols].fillna(fullData[num_cols].mean(),inplace=True)

#Impute categorical missing values with -9999
fullData[cat_cols] = fullData[cat_cols].fillna(value = -9999)

print("After Entering values")
print(fullData.describe())

#create label encoders for categorical features
for var in cat_cols:
 number = LabelEncoder()
 fullData[var] = number.fit_transform(fullData[var].astype('str'))

#Target variable is also a categorical so convert it
#fullData["target"] = number.fit_transform(fullData["target"].astype('float'))

print("___________________________Full Data__________________________")
print(fullData['target'])

train=fullData[fullData['Type']=='Train']
test=fullData[fullData['Type']=='Test']

#train['is_train'] = np.random.uniform(0, 1, len(train)) <= 1.5
#Train, Validate = train[train['is_train']==True], train[train['is_train']==False]

Train = train 
#pass the imputed dummy variable into modelling process

print("pass the imputed dummy variable into modelling process")

features=list(set(list(fullData.columns))-set(target_col)-set(other_col))

print("xtrain and shit")

x_train = Train[list(features)].values

print("####################################X_train##################################")

print(x_train)

y_train = Train["target"].values

print("####################################y_train##################################")

print(y_train)
#x_validate = Validate[list(features)].values
#y_validate = Validate["target"].values
x_test=test[list(features)].values

print("starting random seeeeeed")

random.seed(100)

print("Seeding Done")

rf = RandomForestClassifier(n_estimators=10, n_jobs = -1)

print("Random Forest Function Call done")


rf.fit(x_train, y_train.astype('str'))

print("Random Seeding fitting done")

#status = rf.predict_proba(x_validate)

# fpr, tpr, _ = roc_curve(y_validate, status[:,1])

# print("before auc")
# roc_auc = auc(fpr, tpr)

# print ("post AUC")
# print (roc_auc)

final_status = rf.predict(x_test)

print(final_status)
test["target"]=final_status
print("Final STATUS")
test.to_csv('model_output.csv',columns=['target'])


