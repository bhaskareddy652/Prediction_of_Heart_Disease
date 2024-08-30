import tkinter
from tkinter import *
from tkinter import filedialog
import numpy as np
from PIL import ImageTk
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
import pandas as pd
from string import punctuation
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler as ss
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)
from sklearn.model_selection import train_test_split
# import warnings filter
from warnings import simplefilter
simplefilter(action='ignore', category = FutureWarning)

df = pd.read_csv("C:/Users/bhask/Project/cleveland.csv", header = None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']
### 1 = male, 0 = female
df.isnull().sum()
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['sex'] = df.sex.map({0: 'female', 1: 'male'})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())
main = tkinter.Tk()
main.title("Prediction of Heart Disease Using Machine Learning Algorithms")  # designing main screen
main.geometry("1300x1200") 
global filename
def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset",
                                          title="Select a File",
                                          filetypes=(("Text files",
                                                      "*.csv*"),
                                                     ("all files",
                                                      "*.*")))
    # distribution of target vs age
    sns.set_context("paper", font_scale=2, rc={"font.size": 20, "axes.titlesize": 25, "axes.labelsize": 20})
    sns.catplot(kind='count', data=df, x='age', hue='target', order=df['age'].sort_values().unique())
    plt.title('Variation of Age for each target class')
    plt.show()

    # barplot of age vs sex with hue = target
    sns.catplot(kind='bar', data=df, y='age', x='sex', hue='target')
    plt.title('Distribution of age vs sex with the target class')
    plt.show()

    df['sex'] = df.sex.map({'female': 0, 'male': 1})
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n");

def dataPreprocessing():
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = ss()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    text.delete('1.0', END)
    text.insert(END, X_train, " \n");
    text.insert(END, " \n");
    text.insert(END, " \n");
    text.insert(END, X_test + " \n");
    text.insert(END, " \n");
    text.insert(END, "Data Cleaning Process is completed", "\n");

#global logisticacc, randomforestacc, decisontreeacc, xgbboostacc,naviybayesacc
def logisticsRegression():
    global logisticacc
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import confusion_matrix
    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)
    logisticacc=format((cm_train[0][0] + cm_train[1][1]) / len(y_train))
    print(logisticacc)
    edd='Accuracy for training set for Logistic Regression = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train))
    msgg='Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test))
    text.delete('1.0', END)
    text.insert(END, edd, " \n");
    text.insert(END, " \n");
    text.insert(END, " \n");
    text.insert(END, msgg + " \n");
    text.insert(END, " \n");

def randomForest():
    global randomforestacc
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import confusion_matrix
    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)
    randomforestacc=format((cm_train[0][0] + cm_train[1][1]) / len(y_train))
    print()
    edd='Accuracy for training set for Random Forest = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train))
    msgg='Accuracy for test set for Random Forest = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test))
    text.delete('1.0', END)
    text.insert(END, edd, " \n");
    text.insert(END, " \n");
    text.insert(END, " \n");
    text.insert(END, msgg + " \n");
    text.insert(END, " \n");
def decisionTree():
    global decisontreeacc
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm_test = confusion_matrix(y_pred, y_test)
    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)
    decisontreeacc=format((cm_train[0][0] + cm_train[1][1]) / len(y_train))
    print()
    edd='Accuracy for training set for Decision Tree = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train))
    msgg='Accuracy for test set for Decision Tree = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test))
    text.delete('1.0', END)
    text.insert(END, edd, " \n");
    text.insert(END, " \n");
    text.insert(END, " \n");
    text.insert(END, msgg + " \n");
    text.insert(END, " \n");
def XGBoost():
    global xgbboostacc
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    xg = XGBClassifier()
    xg.fit(X_train, y_train)
    y_pred = xg.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm_test = confusion_matrix(y_pred, y_test)
    y_pred_train = xg.predict(X_train)
    for i in range(0, len(y_pred_train)):
        if y_pred_train[i] >= 0.5:  # setting threshold to .5
            y_pred_train[i] = 1
        else:
            y_pred_train[i] = 0
    cm_train = confusion_matrix(y_pred_train, y_train)
    print()
    xgbboostacc=format((cm_train[0][0] + cm_train[1][1]) / len(y_train))
    edd='Accuracy for training set for XGBoost = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train))
    msgg='Accuracy for test set for XGBoost = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test))
    text.delete('1.0', END)
    text.insert(END, edd, " \n");
    text.insert(END, " \n");
    text.insert(END, " \n");
    text.insert(END, msgg + " \n");
    text.insert(END, " \n");
def NaiveBayes():
    global naviybayesacc
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm_test = confusion_matrix(y_pred, y_test)
    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)
    naviybayesacc=format((cm_train[0][0] + cm_train[1][1]) / len(y_train))
    print()
    edd='Accuracy for training set for Naive Bayes = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train))
    msgg='Accuracy for test set for Naive Bayes = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test))
    text.delete('1.0', END)
    text.insert(END, edd, " \n");
    text.insert(END, " \n");
    text.insert(END, " \n");
    text.insert(END, msgg + " \n");
    text.insert(END, " \n");
def predictDisease():
    import GUI
def algorithmComparision():
    edd1 ="Accuracy for Logistic Regression Algorithm: ="+logisticacc
    edd2 ="Accuracy for Random Forest Algorithm: ="+randomforestacc
    edd3 = "Accuracy for Decision Tree Algorithm: =" + decisontreeacc
    edd4 = "Accuracy for XGBoost Algorithm: =" + xgbboostacc
    edd5 = "Accuracy for Naive Bayes Algorithm: =" + naviybayesacc
    text.delete('1.0', END)
    text.insert(END, edd1+ " \n");
    text.insert(END, " \n");
    text.insert(END, edd2+ " \n");
    text.insert(END, " \n");
    text.insert(END, edd3 + " \n");
    text.insert(END, " \n");
    text.insert(END, edd4 + " \n");
    text.insert(END, " \n");
    text.insert(END, edd5 + " \n");
    text.insert(END, " \n");
font = ('times',16, 'bold')
title = Label(main, text='PREDICTION OF HEART DISEASE USING MACHINE LEARNING ALGORITHMS')
title.config(bg='white', fg='purple')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)
font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Heart Disease Dataset", command=upload)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font1)
readButton = Button(main, text="Data Preprocessing", command=dataPreprocessing)
readButton.place(x=50, y=150)
readButton.config(font=font1)

cleanButton = Button(main, text="Logistic Regression", command=logisticsRegression)
cleanButton.place(x=260, y=150)
cleanButton.config(font=font1)

mlButton = Button(main, text="Random Forest", command=randomForest)
mlButton.place(x=450, y=150)
mlButton.config(font=font1)

mlButton = Button(main, text="Decision Tree", command=decisionTree)
mlButton.place(x=610, y=150)
mlButton.config(font=font1)

mlButton = Button(main, text="XGBoost", command=XGBoost)
mlButton.place(x=750, y=150)
mlButton.config(font=font1)

mlButton = Button(main, text="Naive Bayes", command=NaiveBayes)
mlButton.place(x=850, y=150)
mlButton.config(font=font1)

mlButton = Button(main, text="Results", command=algorithmComparision)
mlButton.place(x=980, y=150)
mlButton.config(font=font1)

mlButton = Button(main, text="Predict Disease", command=predictDisease)
mlButton.place(x=1100, y=150)
mlButton.config(font=font1)
mlButton.config(bg="pink")

font1 = ('times', 12, 'bold')
text = Text(main, height=25, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=200)
text.config(font=font1)

main.config(bg='red')
main.mainloop()
