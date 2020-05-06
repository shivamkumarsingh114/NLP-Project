# coding: utf-8

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
import nltk
from nltk import word_tokenize
import string
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import math
import re
import pickle

_xgb = XGBClassifier(
                      max_depth=7,
                      learning_rate=0.375,
                      n_estimators=110,
                      gamma=0,
                      reg_alpha =0.1,
                      objective = 'multi:softprob',
                      booster='gbtree',
                      silent=True,
                      subsample = .8,
                      colsample_bytree = 0.8,
                      max_delta_step = 1,
                      n_jobs=-1,
                      random_state = 1711
                      )

def clean_text(text):
    text = text.lower()
    to_remove2 = "[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]:"
    table2 = str.maketrans("", "", to_remove2)
    text = text.translate(table2)
    text = text.replace('\r','')
    text = text.replace('\n',' ')
    # text = translator.translate(text).text
    text = text.strip()
    not_required=['a', 'about', 'above', 'after' , 'again' , 'against', 'all', 'am', 'an' , 'and', 'any', 'are',
              'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could',
              'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each', 'few', 'for', 'from', 'further',
              'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself',
              'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it',  'it\'s', 'its', 'itself',
              'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our',
              'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so',
              'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these',
              'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t',
              'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which',
              'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re',
              'you\'ve', 'your', 'yours', 'yourself', 'yourselves']
    tokens = word_tokenize(text)
    result = [i for i in tokens if not i in not_required]
    text = ' '.join(result)
    return text

def CountVec(data):
    count_vec = TfidfVectorizer(
    max_df = 0.3,
    min_df = 3,
    lowercase = True,
    ngram_range = (1,2),
    analyzer = 'word'
    )
    data_count = count_vec.fit_transform(data.replaced_col)
    indices = pd.DataFrame(count_vec.get_feature_names())

    n_comp = 7
    svd_obj = TruncatedSVD(n_components = n_comp, algorithm = 'randomized')
    svd_obj.fit(data_count)
    data_svd = pd.DataFrame(svd_obj.transform(data_count))
    data_svd.columns = ['svd_char_' + str(i) for i in range(n_comp)]
    data = pd.concat([data, data_svd], axis=1)
    del data_count, data_svd
    return data

def replace_words(data):
    regex = re.compile("<(.*?)/>")
    data['replaced_col'] = ""
    for i in range(data.shape[0]):
        data.replaced_col[i] = re.sub(regex, data.edit[i], data.original[i])
    return data

def training_model(data):
    train_labels = data['meanGrade']
    data = data.drop(['meanGrade'], axis = 1)
    print('\nModel Training...\n')
    seed = 7
    test_size = 0.20
    X_train, X_test, y_train, y_test = train_test_split(data, train_labels, test_size=test_size, random_state=seed)
    _xgb.fit(X_train, y_train)
    print('\nTraining Ended...\n')
    predicted_labels = _xgb.predict_proba(X_test)
    pickle.dump(_xgb, open("training-data/task-2/model/xgb_model.sav", 'wb'))
    return predicted_labels, y_test

def score_model(predicted_lables, y_test):
    col =[]
    for i in range(predicted_lables.shape[1]):
      col.append(str(i))
    result = pd.DataFrame(data = predicted_labels, columns = col )
    aa = result.idxmax(axis=1)
    aa=aa.to_frame()
    result_col = aa.values
    result_col = result_col.astype(int)
    score = math.sqrt(mean_squared_error(y_test, result_col))
    return score

def judges_cols(data):
    for i in range(1,6):
        data['judge'+str(6-i)] = data['grades'].apply(lambda x : (int)(x%10))
        data['grades'] = data['grades'].apply(lambda x : (int)(x/10))
    data = data.drop(['grades'], axis=1)
    return data


if __name__ == '__main__':

    data_path = "training-data/task-1/train_funlines.csv"
    data = pd.read_csv(data_path)
    data = replace_words(data)
    data = data.drop(['id', 'original', 'edit'], axis=1)
    data['replaced_col'] = data['replaced_col'].apply(lambda x : clean_text(x))
    data = CountVec(data)
    data=data.drop(['replaced_col'], axis=1)
    data = judges_cols(data)
    print(data.head())
    predicted_labels, y_test = training_model(data)
    score = score_model(predicted_labels, y_test)
    print("root mean square error for model: " + str(score))
