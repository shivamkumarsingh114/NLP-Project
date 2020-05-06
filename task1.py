import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from scipy.sparse import hstack
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
import argparse

class LinReg():
    """Train, Test and Visualise a Linear Reg Model"""

    def __init__(self, data_loc):
        self.data_loc = data_loc

    def train_data(self, loc = None ,split = 0.1) :
        data = pd.read_csv(self.data_loc)
        data["original"].str.lower()
        data["edit"].str.lower()

        feature_cols = ["original", "edit"]
        X = data[feature_cols]
        y = data[["meanGrade"]]
        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=split)
        vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2',
                                    ngram_range=(1, 2), stop_words='english')
        X_train_o = vectorizer.fit_transform(X_train["original"], X_train["edit"])
        X_test_o = vectorizer.transform(X_test["original"], X_test["edit"])

        linreg = LinearRegression()
        linreg.fit(X_train_o, y_train)
        pred = linreg.predict(X_test_o)

        lis1 = y_test["meanGrade"].values.tolist()
        print_data = X_test
        print_data["meanGrade"] = lis1
        print_data["PredGrade"] = pred
        lis2 = print_data.values.tolist()

        for i in lis2:
            print("original:",i[0], "Edited Word:", i[1], "Mean Grade:",i[2], "Prediction:",i[3])
        print('Mean squared error: %.2f'
              % sqrt(mean_squared_error(y_test, pred)))

        filename1 = loc + 'finalized_model.sav'
        fileneame2 = loc + "feature.pkl"
        pickle.dump(linreg, open(filename1, 'wb'))
        pickle.dump(vectorizer.vocabulary_,open(fileneame2,"wb"))

    def test_data(self, loc):
            filename1 = loc + 'finalized_model.sav'
            fileneame2 = loc + "feature.pkl"
            data = pd.read_csv(self.data_loc)
            data["original"].str.lower()
            data["edit"].str.lower()

            feature_cols = ["original", "edit"]
            X = data[feature_cols]
            y = data[["meanGrade"]]

            transformer = TfidfTransformer()
            vectorizer = TfidfVectorizer(decode_error="replace",
                                vocabulary=pickle.load(open(fileneame2, "rb")))
            X_o = vectorizer.fit_transform(X["original"], X["edit"])
            linreg = pickle.load(open(filename1, "rb"))
            pred = linreg.predict(X_o)
            lis1 = y["meanGrade"].values.tolist()
            print_data = X
            print_data["meanGrade"] = lis1
            print_data["PredGrade"] = pred
            lis2 = print_data.values.tolist()

            for i in lis2:
                print("original:",i[0], "Edited Word:", i[1], "Mean Grade:",i[2], "Prediction:",i[3])
                # print("original:",i[0], "Edited Word:", i[1], "Prediction:",i[2])
            print('Mean squared error: %.2f'
                  % sqrt(mean_squared_error(y, pred)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-tr", "--train", dest = "tr", help="Training Mode")
    parser.add_argument("-te", "--test", dest = "te", help="Testing Mode")
    parser.add_argument("-m", "--model", dest = "m", help="Path of Model")
    parser.add_argument("-ds", "--dataset", dest = "ds", help="Path of Dataset")
    parser.add_argument("-tts", "--train_test_split", dest = "tts", help="Split Ratio")

    args = parser.parse_args()
    if args.tr != None:
        try:
            data = args.ds
            model = args.m
            lr = LinReg(data)
        except:
            print("Format is: -ds ./data/file -m ./model/ -tts optional")
        lr.train_data(model, args.tts)
    elif args.te != None:
        try:
            data = args.ds
            model = args.m
            lr = LinReg(data)
        except:
            print("Format is: -ds ./data/file -m ./model/")
        lr.test_data(model)
    else:
        print("Oops, invalid command!")
