import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
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
from gensim.models import Word2Vec

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

def replace_words(data):
    regex = re.compile("<(.*?)/>")
    data['replaced_col'] = ""
    for i in range(data.shape[0]):
        data.replaced_col[i] = re.sub(regex, data.edit[i], data.original[i])
    return data

def vectorize_sentences(data):
    text=[]
    for sentence in data['replaced_col']:
      sent_word_list = [word for word in sentence.lower().split()]
      text.append(sent_word_list)

    w2v = Word2Vec(text, min_count=1)
    vect_record=[]
    for i in range(len(data['replaced_col'])):
          sent = data.replaced_col[i]
          if len(sent)!=0:
            sent_vect = sum([w2v[w] for w in sent.lower().split()])/(len(sent.split())+0.001)
          else:
            sent_vect = np.zeros((100,))
          vect_record.append(sent_vect)

    return vect_record

def training_model(X,y):
    print('\nModel Training...\n')
    seed = 7
    test_size = 0.20
    label_encoded_y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, label_encoded_y, test_size=test_size, random_state=seed)
    _xgb.fit(X_train, y_train)
    print('\nTraining Ended...\n')
    predicted_labels = _xgb.predict_proba(X_test)
    pickle.dump(_xgb, open("training-data/task-2/model/xgb_word2vec_model.sav", 'wb'))
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
    score = mean_squared_log_error(y_test, result_col)
    return score

if __name__ == '__main__':

    data_path = "training-data/task-1/train_funlines.csv"
    data = pd.read_csv(data_path)
    data = replace_words(data)
    data = data.drop(['id', 'original', 'edit'], axis=1)
    data['replaced_col'] = data['replaced_col'].apply(lambda x : clean_text(x))
    vect_record = vectorize_sentences(data)
    X = pd.DataFrame(vect_record, columns=range(100))
    y = data['meanGrade']
    predicted_labels, y_test = training_model(X,y)
    score = score_model(predicted_labels, y_test)
    print("mean square log error for model: " + str(score))
