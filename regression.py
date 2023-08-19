import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
np.set_printoptions(precision=3, suppress=True)

########################################################################################################################
### Regression #########################################################################################################
########################################################################################################################
# Linear Regression, Multiple Linear Regression and Deep Neural Network

### Settings ###########################################################################################################
SETTING_FILENAME = "data/Life Expectancy Data.csv" # https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
SETTING_LABELCOLUMN = "Life expectancy"

### Functions ##########################################################################################################
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)

def plot_labelprediction(examples_feature, examples_label, predict_x, predict_y):
  # examples_feature, examples_label is a pd series
  plt.scatter(examples_feature, examples_label, label='Data')
  plt.plot(predict_x, predict_y, color='k', label='Predictions')
  plt.xlabel(examples_feature.name)
  plt.ylabel(SETTING_LABELCOLUMN)
  plt.legend()

### Read Data ##########################################################################################################
df = pd.read_csv(SETTING_FILENAME)

### Data Cleanup #######################################################################################################
# rename columns (strip whitespace)
df.rename(columns={x: x.strip() for x in df.columns}, inplace=True)

# replace nan values with mean
for i in df.columns[df.isnull().any(axis=0)]:
    df[i].fillna(df[i].mean(), inplace=True)
# df = df.dropna()
"""
print(df.head())
print(df.describe().T)
print("number nan values:", df.isna().sum())
"""

### Feature Engineering ################################################################################################
# convert categorical columns
df = pd.get_dummies(df, dtype=float)

### Create Dataset #####################################################################################################
train_dataset = df.sample(frac=0.9, random_state=0)
test_dataset = df.drop(train_dataset.index)

train_features = train_dataset.copy()
train_labels = train_features.pop(SETTING_LABELCOLUMN)
test_features = test_dataset.copy()
test_labels = test_features.pop(SETTING_LABELCOLUMN)

### Models #############################################################################################################
performance = {}

# Linear Regression
for feature in (x for x in df.columns if x != SETTING_LABELCOLUMN):
    normalizer = keras.layers.Normalization(input_shape=[1,], axis=None,)
    normalizer.adapt(np.array(train_features[feature]))
    model_lr = keras.Sequential([normalizer, keras.layers.Dense(units=1)])
    model_lr.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')
    history = model_lr.fit(train_features[feature], train_labels, epochs=6, validation_split=0.2, verbose=1)
    performance["lr_" + feature.lower().replace("", "")] = model_lr.evaluate(test_features[feature], test_labels)

# Multiple Linear Regression
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
model_mlr = tf.keras.Sequential([normalizer,keras.layers.Dense(units=1)])
model_mlr.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')
history = model_mlr.fit(train_features, train_labels, epochs=16, validation_split=0.2, verbose=1)
performance["mlr"] = model_mlr.evaluate(test_features, test_labels)

# Deep Neural Network
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
model_dnn = keras.Sequential([
       normalizer,
       keras.layers.Dense(64, activation='relu'),
       keras.layers.Dense(64, activation='relu'),
       keras.layers.Dense(1)])
model_dnn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error')
history = model_dnn.fit(train_features, train_labels, epochs=16, validation_split=0.2, verbose=1)
performance["dnn"] = model_dnn.evaluate(test_features, test_labels)

# => Performance
print(pd.DataFrame(performance, index=["MAE"]).T.sort_values(by="MAE"))

"""
# plot examples and prediction
feature = "Year"
predict_x = tf.linspace(min(train_features[feature]), max(train_features[feature]), max(train_features[feature]) - min(train_features[feature]))
predict_y = model_xy.predict(predict_x)
plot_labelprediction(train_features[feature], train_labels, predict_x, predict_y) 
plt.show()
"""
"""
# plot loss and val_loss
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail()) 
plot_loss(history) 
plt.show()
"""