import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.metrics import mean_squared_error

sentences=[line.split(',,,')[0].strip() for line in open("LabelledData.txt").readlines()]
responses=[line.split(',,,')[1][:-1].strip() for line in open("LabelledData.txt").readlines()]
test_sentences=[(' ').join(line.split(' ')[1:]).strip() for line in open("test.txt").readlines()]

print(pd.Series(responses).value_counts())

replace_index={'what':0,'who':1,'unknown':2,'affirmation':3,'when':4}
y_train=pd.Series(responses).replace(to_replace=replace_index)


sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True)
sklearn_representation = sklearn_tfidf.fit_transform(sentences)
train=pd.DataFrame(sklearn_representation.todense())
train['y']=y_train
test=pd.DataFrame(sklearn_tfidf.transform(test_sentences).todense())

#create validation set
x_train=train.sample(frac=0.1, replace=False)
x_valid=pd.concat([train, x_train]).drop_duplicates(keep=False)
x_test=test

dtrain = xgb.DMatrix(x_train.ix[:, x_train.columns != 'y'], x_train['y'], missing=np.nan)
dvalid = xgb.DMatrix(x_valid.ix[:, x_valid.columns != 'y'], x_valid['y'], missing=np.nan)
dtest = xgb.DMatrix(x_test, missing=np.nan)

nrounds = 150
watchlist = [(dtrain, 'train')]

params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.09, "max_depth": 10, "subsample": 0.9, "colsample_bytree": 0.75,
                "min_child_weight": 1, "n_estimators":50,"num_class": 5,
                "seed": 2016, "tree_method": "exact"}

bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
valid_preds=bst.predict(dvalid)
print(mean_squared_error(list(x_valid['y']), valid_preds))

test_preds = bst.predict(dtest)
replace_index_rev={0:'what',1:'who',2:'unknown',3:'affirmation',4:'when'}
test_preds_labels=pd.Series(test_preds).replace(to_replace=replace_index_rev)
submit_test = pd.DataFrame({'sentences': test_sentences, 'labels': test_preds_labels})
submit_test.to_csv("submit_test.csv",sep='$',index=False)

