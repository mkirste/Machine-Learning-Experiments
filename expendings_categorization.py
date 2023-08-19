import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

########################################################################################################################
### Expendings Categorization ##########################################################################################
########################################################################################################################
# input is a table with rows for expendings and the following columns:
#   Feature: "Buchungstag" (bookingday) [DD.MM.YYYY format]
#   Feature: "Verwendungszweck" (purpose) [string]
#   Feature: "Betrag" (amount) [float]
#   Label: "Kategorie" (category) [string]
# => rows without labels will get predicted, word dictionary for words in "Verwendungszweck" will be automatically created
# => outputs unlabeled rows with category_prediction_index, category_prediction_value, and prediction_probability


### Settings ###########################################################################################################
SETTING_INPUTFILENAME = "data/expendings.csv"
SETTING_OUTPUTFILENAME = "data/expendings_predictions.csv"
SETTING_SEP = ";"
# prediction
SETTING_PREDICTIONMETHOD = 'decisiontree' # machinelearning, decisiontree, or randomforest 
PREDICTION_PROBABILITY = 0.5 # defines 
# word dictionary
MIN_WORD_OCCURENCES = 4 # min number occurrences for words to be included in word dictionary
MIN_WORD_LENGTH = 4 # min word length
MAX_LENGTH_WORD_DICTIONARY = 1000 # max number words in dictionary

### Functions ##########################################################################################################
def PlotTfkerasModelLearningCurves(history, parameter):
    """
    Plot learning curves from 'history' object.

    Parameters:
        history (History object): result of the `fit` method execution
        parameter (str): parameter to explore: 'accuracy', 'loss', etc.
    """

    plt.plot(history.history[parameter])
    plt.plot(history.history["val_" + parameter])
    plt.xlabel("Epochs")
    plt.ylabel(parameter)
    plt.legend([parameter, "val_" + parameter])
    plt.show()

def GetWordCountDictionary(str):
    """
    Splits the text and returns a dictionary with all words and their respective count

    Parameters
    str: a string with text input
    Returns
    dictionary with words and number occurences
    """
    words = {}
    for word in str.split():
        if word in words:
            words[word] += 1
        else:
            words[word] = 1
    return words

def GetSortedWordList(series, min_word_occurences=4, min_word_length=4):
    """
    Creates a word dictionary

    Parameters
    series: a panda string series
    minimum_number_occurences: min number of word occurences to appear in result
    minimum_word_length: min word length to appear in result
    Returns
    sorted list with words and number occurences tuple
    """
    content = series.to_string(index=False)
    content = content.replace(":", " ").replace(".", " ").replace(",", " ").replace("/", " ")
    content = content.replace("(", "").replace(")", "").replace("-", "")

    list = GetWordCountDictionary(content).items()
    list = filter(lambda x: len(x[0]) >= min_word_length, list)  # filter by minimum word length
    list = filter(lambda x: x[1] >= min_word_occurences, list)  # filter by minimum number occurences
    list = sorted(list, key=lambda x: x[1], reverse=True)  # sort by number occurences

    return list

def AdaptInputData(df, min_word_occurences=4, min_word_length=4, max_length_word_dictionary=1000, normalize_data=True):
    df = df.copy()

    # 1. Data Cleaning #########################################################
    # rename columns
    df.rename(columns={"Buchungstag": "bookingday", "Verwendungszweck": "purpose", "Betrag": "amount", "Kategorie": "category",}, inplace=True)

    # 2. Feature Selection/Engineering #########################################
    # create bookingday_month and bookingday_day feature, drop bookingday feature
    df["bookingday"] = pd.to_datetime(df["bookingday"], format="%d.%m.%Y")
    df["bookingday_month"] = df["bookingday"].dt.month
    df["bookingday_day"] = df["bookingday"].dt.day
    df.drop(columns=["bookingday"], inplace=True)

    # create one-hot word features for purpose feature
    df["purpose"].fillna("", inplace=True)
    df["purpose"] = df["purpose"].str.lower()
    df["purpose"] = df["purpose"].apply(lambda x: re.sub('[^a-z ]', "", x))
    purpose_wordlist = GetSortedWordList(df["purpose"], min_word_occurences, min_word_length)
    for word, number_occurences in purpose_wordlist[:max_length_word_dictionary]:  # create word columns
        new_series = df["purpose"].str.contains(word, regex=False)
        new_series = new_series.astype("int8")
        df["purpose_" + word] = new_series
        if df.shape[1] % 50 == 0:
            df = df.copy()
    df.drop(columns=["purpose"], inplace=True)

    # amount feature
    df["amount"] = df["amount"].str.replace(",", ".").astype("float32")

    # create category label
    df["category"] = df["category"].astype("category")
    category_dictionary = dict(enumerate(df["category"].cat.categories))  # dict(zip(df['category'].cat.codes, df['category']))    
    df["category"] = pd.Categorical(df["category"].cat.codes, categories=list(category_dictionary.keys())) # df["category"] = df["category"].cat.codes 

    # adapt column order
    col_list = df.columns.tolist()
    cat_index = df.columns.get_loc("category")
    new_cols_order = (col_list[0:cat_index] + col_list[cat_index + 1 :] + col_list[cat_index : cat_index + 1])
    df = df[new_cols_order]

    # 3. Data Normalization ####################################################
    if normalize_data:
        # amount bookingday_month  bookingday_day
        df["amount"] = (df["amount"] - df["amount"].mean()) / df["amount"].std()
        # df["amount"] == (df["amount"] - df["amount"].min()) / (df["amount"].max() - df["amount"].min())
        df["bookingday_month"] = df["bookingday_month"] / df["bookingday_month"].max()
        df["bookingday_day"] = df["bookingday_day"] / df["bookingday_day"].max()

    return df, category_dictionary

def TransformPredictions(Y_prediction_probabilities, category_dictionary):
    category_indices = np.argmax(Y_prediction_probabilities.T, 0)
    category_values = np.vectorize(category_dictionary.get)(category_indices)
    probabilities = Y_prediction_probabilities[
        [x for x in range(Y_prediction_probabilities.shape[0])], category_indices
    ]

    Y_prediction_df = pd.DataFrame({
            "category_prediction_index": category_indices,
            "category_prediction_value": category_values, 
            "prediction_probability": probabilities
            })

    return (Y_prediction_probabilities, Y_prediction_df)

def PrintPredictionAccuracy(Y_prediction_df, Y, prediction_probability=0.5):
    Y_prediction = Y_prediction_df["category_prediction_index"].values
    Y_probabilities = Y_prediction_df["prediction_probability"].values

    result_all = Y_prediction == Y
    print("accuracy all examples ({}): {:.2f}".format(result_all.size, result_all.sum() / result_all.size))

    confidend_prediction_indices = np.argwhere(Y_probabilities >= prediction_probability)
    result_confident = (Y[confidend_prediction_indices] == Y_prediction[confidend_prediction_indices])
    print("accuracy examples with prediction probability >= {} ({}): {:.2f}".format(prediction_probability, result_confident.size, result_confident.sum() / result_confident.size))

def CreatePredictionDataFrame(df_input, Y2_prediction_df):
    df_input_unlabeled = df_input[df_input["Kategorie"].isna()]
    df_output = pd.concat([df_input_unlabeled.reset_index(drop=True), Y2_prediction_df.reset_index(drop=True),], axis=1)
    return df_output

### Read Data ##########################################################################################################
df_input = pd.read_csv(SETTING_INPUTFILENAME, sep=SETTING_SEP)

### Adapt Data #########################################################################################################
df, category_dictionary = AdaptInputData(df_input, MIN_WORD_OCCURENCES, MIN_WORD_LENGTH, MAX_LENGTH_WORD_DICTIONARY)
# print("df:", df.shape, df.dtypes, df.head(10), df.describe().T, "categories:", df["category"].unique(), category_dictionary)

df_labeled = df[df["category"].notnull()]
df_unlabeled = df[df["category"].isnull()]
#df_labeled = df_labeled.sample(frac=1) # shuffle dataset
#df_labeled = df_labeled[0:100] # select first x entries

X = np.array(df_labeled)[:, 0:-1]
Y = np.array(df_labeled)[:, -1].astype("int")
Y_onehot = pd.get_dummies(df_labeled["category"], prefix=['cat'], dtype=int)
X2 = np.array(df_unlabeled)[:, 0:-1]

### Train Model ########################################################################################################
match SETTING_PREDICTIONMETHOD:
    case 'machinelearning':
        model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=len(category_dictionary) * 8, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(units=len(category_dictionary) * 4, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(units=len(category_dictionary) * 2, activation="relu"),
                tf.keras.layers.Dense(len(category_dictionary), activation="softmax"),
            ])
        model.compile(optimizer="adam", loss="categorical_crossentropy") # sparse_categorical_crossentropy, if y is not one-hot encoded
        history = model.fit(x=X, y=Y_onehot, validation_split=0.2, epochs=12)
        # print("evaluation:", model.evaluate(X, Y_onehot))
        # PlotTfkerasModelLearningCurves(history, "loss") # loss, accuracy
        Y_prediction_probabilities, Y_prediction_df = TransformPredictions(model.predict(X), category_dictionary)

    case 'decisiontree' | 'randomforest':
        if SETTING_PREDICTIONMETHOD == 'decisiontree':        
            model = tree.DecisionTreeClassifier(max_depth=16, min_samples_leaf=4) # Decision Tree
            # tree.plot_tree(model)
            # plt.show()
        else:
            model = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=4) # Random Forest        
        model = model.fit(X, Y_onehot)
        model_predictions = list(map(lambda x: (x if x.shape[1] == 2 else np.pad(x, ((0,0),(0,1)))), [x for x in model.predict_proba(X)]))        
        Y_prediction_probabilities, Y_prediction_df = TransformPredictions(np.array(model_predictions)[:, :,1].T, category_dictionary)
        
PrintPredictionAccuracy(Y_prediction_df, Y, PREDICTION_PROBABILITY)

### Make Predictions ###################################################################################################
Y2_prediction_probabilities, Y2_prediction_df = TransformPredictions(model.predict(X2), category_dictionary)
df_input_unlabeled = df_input[df_input["Kategorie"].isna()]
df_output = CreatePredictionDataFrame(df_input, Y2_prediction_df)
df_output.to_csv(SETTING_OUTPUTFILENAME, sep=SETTING_SEP, decimal=",", float_format="%.2f")
