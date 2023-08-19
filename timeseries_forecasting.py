import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

########################################################################################################################
### Timeseries Forecasting #############################################################################################
########################################################################################################################
# input requires table with rows for time steps and columns for signals (plus one time column)
# (acknowledge to Tensor Flow time series forecasting tutorial)

### Settings ###########################################################################################################
# data (rows => time steps, columns => signals (features/labels))
SETTING_FILENAME = "data/jena_climate_2009_2016.csv" # https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip
SETTING_TIMECOLUMN_NAME = "Date Time"
SETTING_TIMECOLUMN_FORMAT = "%d.%m.%Y %H:%M:%S"
SETTING_SKIPSTEPS = 6 # None or Integer
# input steps and output steps (time window)
SETTING_INPUTSTEPS = 24 # number input steps for prediction [labels]
SETTING_SHIFTSTEPS = 0
SETTING_OUTPUTSTEPS = 24 # number output steps (predictions) [features => prediction]
# input labels and output features (signals)
SETTING_INPUTSIGNALS = ["p (mbar)", "T (degC)", "Wx", "Wy", "Day sin", "Day cos", "Year sin", "Year cos"] # None for all [labels]
SETTING_OUTPUTSIGNALS = ["T (degC)"] # None for all [features => prediction]
# train/validation split
SETTING_TRAINSPLIT = 0.7 # between 0 and 1
SETTING_VALSPLIT = 0.2  # between 0 and 1
# models
SETTING_MAX_EPOCHS = 1
SETTING_LOSS = tf.keras.losses.MeanSquaredError()
SETTING_METRICS = [tf.keras.metrics.MeanAbsoluteError()]


### Read Data ##########################################################################################################
df = pd.read_csv(SETTING_FILENAME) # (time, signals)
if SETTING_SKIPSTEPS != None:
    df = df[SETTING_SKIPSTEPS-1::SETTING_SKIPSTEPS]
date_time = pd.to_datetime(df.pop(SETTING_TIMECOLUMN_NAME), format=SETTING_TIMECOLUMN_FORMAT)

### Data Cleanup #######################################################################################################
# error in wv and max wv columns
df["wv (m/s)"][df["wv (m/s)"] == -9999.0] = 0.0 # error in wv
df["max. wv (m/s)"][df["max. wv (m/s)"] == -9999.0] = 0.0 # error in max wv

### Feature Engineering ################################################################################################
# convert wind direction and velocity to wind vector
wv = df.pop("wv (m/s)")
max_wv = df.pop("max. wv (m/s)")
wd_rad = df.pop("wd (deg)")*np.pi / 180 # wd as radians
df["Wx"] = wv*np.cos(wd_rad)
df["Wy"] = wv*np.sin(wd_rad)
df["max Wx"] = max_wv*np.cos(wd_rad)
df["max Wy"] = max_wv*np.sin(wd_rad)
# create "Time of day" and "Time of year" signals
timestamp_s = date_time.map(pd.Timestamp.timestamp) # datetime as timestamp
df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / (24*60*60)))
df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / (24*60*60)))
df["Year sin"] = np.sin(timestamp_s * (2 * np.pi / (365.2425*24*60*60)))
df["Year cos"] = np.cos(timestamp_s * (2 * np.pi / (365.2425*24*60*60)))
"""
print(df.head())
print(df.describe().T)
"""

### Create Dataset #####################################################################################################
class Dataset():
  def __init__(self, df, input_steps, shift_steps, output_steps, input_signals = None, output_signals = None, train_split = 0.7, val_split = 0.2, normalize_data = True):
    """
    Create Dataset (steps => rows, signals => columns)

    Parameters:
        input_steps, output_steps, shift_steps: define time window for input/output time steps (input_steps=4, shift_steps=1, output_steps=2 => [0,1,3,4] [6,7])
        input_signals: input columns (list)
        output_signals: output columns (list)
    """
    # create df
    self.df = df
    if len(input_signals) != len(df.columns):
        self.df = df[df.columns.intersection(input_signals)]
    self.df_num_steps = self.df.shape[0] # len(self.df)
    self.df_num_signals = self.df.shape[1]

    # signals
    input_signals = list(self.df.columns) if input_signals == None else input_signals
    output_signals = list(self.df.columns) if output_signals == None else output_signals
    self.input_signals = list(set(input_signals) | set(output_signals)) 
    self.output_signals = output_signals
    self.num_input_signals = len(self.input_signals)
    self.num_output_signals = len(self.output_signals)
    self.df_signal_index_dic = {name: i for i, name in enumerate(self.df.columns)}
    self.df_input_indices = list(self.df_signal_index_dic.values())
    self.df_output_indices = [self.df_signal_index_dic[x] for x in self.output_signals]

    # split data
    val_start_index = int(self.df_num_steps * train_split)
    test_start_index = int(self.df_num_steps * (train_split + val_split))
    self.df_train = self.df[0:val_start_index]
    self.df_val = self.df[val_start_index:test_start_index]
    self.df_test = self.df[test_start_index:]

    # normalize data
    if normalize_data == True:
        train_mean = self.df_train.mean()
        train_std = self.df_train.std()
        self.df_train = (self.df_train - train_mean) / train_std
        self.df_val = (self.df_val - train_mean) / train_std
        self.df_test = (self.df_test - train_mean) / train_std

    # steps (time window)
    self.input_steps = input_steps
    self.shift_steps = shift_steps
    self.output_steps = output_steps
    self.total_steps = input_steps + shift_steps + output_steps

    self.input_slice = slice(0, self.input_steps) # [0:input_steps]
    self.input_indices = np.arange(self.total_steps)[self.input_slice]
    self.output_slice = slice(self.input_steps + self.shift_steps, None) # [input_steps+shift_steps-label_width:]
    self.output_indices = np.arange(self.total_steps)[self.output_slice]

  def __repr__(self):
    return '\n'.join([
        f'Dataset',  
        f'Number rows (time steps): {self.df_num_steps}',
        f'Number signals (signals): {self.df_num_signals}',
        f'Input signals [features]: {self.input_signals}',
        f'Output signals [labels]: {self.output_signals}',
        f'Time steps [input_steps={self.input_steps}, shift_steps={self.shift_steps}, output_steps={self.output_steps}]',
        f'Total steps (window size): {self.total_steps}',
        f'Input step indices [features]: {self.input_indices}',
        f'Output step indices [labels]: {self.output_indices}'])

  def create_tf_dataset(self, df):  # df (time, signals) => x * input (batch, time, signals) label (batch, time, signals) tuple
    data = np.array(df, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_steps, sequence_stride=1, shuffle=True, batch_size=32) # tf.data.Dataset
    # batch_size=None => x * (time, signals)[sequence_length, number signals]
    # batch_size=32   => x * (batch/examples,, time, signals)[batch_size, sequence_length, number signals]
    ds = ds.map(self.apply_windowsplit) # create input label tuples
    return ds
  
  def apply_windowsplit(self, data): # (batch, time, signals)
    inputs = data[:, self.input_slice, :] # features
    outputs = data[:, self.output_slice, :] # labels
    if len(self.output_signals) != len(self.df.columns):
      outputs = tf.stack([outputs[:, :, self.df_signal_index_dic[name]] for name in self.output_signals], axis=-1)
    inputs.set_shape([None, self.input_steps, None])
    outputs.set_shape([None, self.output_steps, None])
    return inputs, outputs # (batch, time, signals), (batch time, signals)

  def plot_example(self, model=None, column_name=None, max_subplots=3, show_plot=True):
      self.plot(self.example, model, column_name, max_subplots, show_plot)

  def plot(self, data, model=None, column_name=None, max_subplots=3, show_plot=True):
    inputs, outputs = data # tf input output
    num_examples = min(max_subplots, len(inputs))
    if column_name == None:
        column_name = self.output_signals[0]
    column_index = self.df_signal_index_dic[column_name]
    
    plt.figure(figsize=(12, 8))
    for e in range(num_examples):
      plt.subplot(num_examples, 1, e+1)
      plt.ylabel(f'{column_name} [normed]')
      plt.plot(self.input_indices, inputs[e, :, column_index], label='Inputs', marker='.', zorder=-10) # features
      output_signals_indices = {name: i for i, name in enumerate(self.output_signals)}
      output_signal_index = output_signals_indices.get(column_name, None)
      plt.scatter(self.output_indices, outputs[e, :, output_signal_index], edgecolors='k', label='Outputs', c='#2ca02c', s=64) # labels
      if model is not None:
        predictions = model(inputs)
        plt.scatter(self.output_indices, predictions[e, :, output_signal_index], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64) # output predictions
      if e == 0:
        plt.legend()
    plt.xlabel('Time') # steps
    if show_plot:
      plt.show()

  @property
  def train(self):
    return self.create_tf_dataset(self.df_train)

  @property
  def val(self):
    return self.create_tf_dataset(self.df_val)

  @property
  def test(self):
    return self.create_tf_dataset(self.df_test)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels`"""
    result = getattr(self, '_example', None)
    if result is None:
      result = next(iter(self.train))
      self._example = result
    return result # input (batch, time, signals) label (batch, time, signals) tuple
  
dataset = Dataset(df, SETTING_INPUTSTEPS, SETTING_SHIFTSTEPS, SETTING_OUTPUTSTEPS, SETTING_INPUTSIGNALS, SETTING_OUTPUTSIGNALS, SETTING_TRAINSPLIT, SETTING_VALSPLIT, True)

"""
print(dataset)
print(dataset.df.head())
print(dataset.df.describe().T)
"""
"""
dataset.plot_example(None)
"""

### Machine Learning Models ############################################################################################
def compile_and_fit(model, dataset, patience=2):
  print("Compile and fit model")
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
  model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(), metrics=SETTING_METRICS)
  history = model.fit(dataset.train, epochs=SETTING_MAX_EPOCHS, validation_data=dataset.val, callbacks=[early_stopping])
  return history

class Baseline(tf.keras.Model):
  def __init__(self, output_indices):
    super().__init__()
    self.output_indices = output_indices

  def call(self, inputs):
    result = tf.tile(inputs[:, -1:, :], [1, dataset.output_steps, 1])
    result = tf.gather(result, indices=self.output_indices, axis=-1)
    return result

class BaselineRepeat(tf.keras.Model):
  def __init__(self, input_steps, output_steps, output_indices):
    super().__init__()
    self.input_steps = input_steps
    self.output_steps = output_steps
    self.output_indices = output_indices

  def call(self, inputs):
    result = inputs
    if(self.output_steps <= self.input_steps):
      result = result[:, -self.output_steps:, :] # result[:, 0:dataset.output_steps, :]
    else:
      time_repeat = self.output_steps // self.input_steps + (1 if self.output_steps % self.input_steps else 0)
      result = tf.tile(result[:, :, :], [1, time_repeat, 1])
      result = result[:, 0:dataset.output_steps, :]
    result = tf.gather(result, indices=self.output_indices, axis=-1)
    return result


performance_val = {}
performance_test = {}

# Baseline Model (prediction is last value)
model_baseline = Baseline(dataset.df_output_indices)
model_baseline.compile(loss=SETTING_LOSS,metrics=SETTING_METRICS)
performance_val["Baseline"] = model_baseline.evaluate(dataset.val)
performance_test["Baseline"] = model_baseline.evaluate(dataset.test, verbose=0)

# Baseline Model Repeat (prediction are last values)
model_baselinerepeat = BaselineRepeat(dataset.input_steps,dataset.output_steps, dataset.df_output_indices)
model_baselinerepeat.compile(loss=SETTING_LOSS,metrics=SETTING_METRICS)
performance_val["BaselineRepeat"] = model_baselinerepeat.evaluate(dataset.val)
performance_test["BaselineRepeat"] = model_baselinerepeat.evaluate(dataset.test, verbose=0)

# Linear Model (prediction is linear to features of single prev time step)
model_linear = tf.keras.Sequential([ # Shape [batch, time, signals]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]), # Shape => [batch, 1, signals] (takes only last time step of input)
    tf.keras.layers.Dense(dataset.output_steps*dataset.num_output_signals, kernel_initializer=tf.initializers.zeros()), # Shape => [batch, 1, output_steps*num_output_signals]
    tf.keras.layers.Reshape([dataset.output_steps, dataset.num_output_signals]) # Shape => [batch, output_steps, num_output_signals]
])
history = compile_and_fit(model_linear, dataset)
performance_val['Linear'] = model_linear.evaluate(dataset.val)
performance_test['Linear'] = model_linear.evaluate(dataset.test, verbose=0)

# Linear Model 2 
model_linear2 = tf.keras.Sequential([ # Shape [batch, time, signals]
    tf.keras.layers.Flatten(), # Shape => [batch, output_steps*num_output_signals]
    tf.keras.layers.Dense(dataset.output_steps*dataset.num_output_signals, kernel_initializer=tf.initializers.zeros()), # Shape => [batch, 1, output_steps*num_output_signals]
    tf.keras.layers.Reshape([dataset.output_steps, dataset.num_output_signals]) # Shape => [batch, output_steps, num_output_signals]
])
history = compile_and_fit(model_linear2, dataset)
performance_val['Linear2'] = model_linear2.evaluate(dataset.val)
performance_test['Linear2'] = model_linear2.evaluate(dataset.test, verbose=0)

# Dense Model
model_dense = tf.keras.Sequential([ # Shape [batch, time, signals]
    tf.keras.layers.Flatten(), # Shape => [batch, output_steps*num_output_signals]
    # tf.keras.layers.Dense(1048, activation='relu'), # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'), # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(dataset.output_steps*dataset.num_output_signals, kernel_initializer=tf.initializers.zeros()), # Shape => [batch, 1, output_steps*num_output_signals]
    tf.keras.layers.Reshape([dataset.output_steps, dataset.num_output_signals]) # Shape => [batch, output_steps, num_output_signals]
])
history = compile_and_fit(model_dense, dataset)
performance_val['Dense'] = model_dense.evaluate(dataset.val)
performance_test['Dense'] = model_dense.evaluate(dataset.test, verbose=0)

# CNN Model (multiple prev time steps => CONV_WIDTH)
CONV_WIDTH = 3
model_conv = tf.keras.Sequential([ # Shape [batch, time, signals]
    tf.keras.layers.Lambda(lambda x: x[:, -min(dataset.input_steps, CONV_WIDTH):, :]), # Shape => [batch, CONV_WIDTH, features]   
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=min(dataset.input_steps, CONV_WIDTH)), # Shape => [batch, 1, conv_units]    
    tf.keras.layers.Dense(dataset.output_steps*dataset.num_output_signals, kernel_initializer=tf.initializers.zeros()), # Shape => [batch, 1, output_steps*num_output_signals]
    tf.keras.layers.Reshape([dataset.output_steps, dataset.num_output_signals]) # Shape => [batch, output_steps, num_output_signals]
])
history = compile_and_fit(model_conv, dataset)
performance_val['Convolution'] = model_conv.evaluate(dataset.val)
performance_test['Convolution'] = model_conv.evaluate(dataset.test, verbose=0)

# RNN Model
model_lstm = tf.keras.Sequential([  # Shape [batch, time, signals]
    tf.keras.layers.LSTM(32, return_sequences=False), # Shape [batch, lstm_units]
    tf.keras.layers.Dense(dataset.output_steps*dataset.num_output_signals, kernel_initializer=tf.initializers.zeros()), # Shape => [batch, 1, output_steps*num_output_signals]
    tf.keras.layers.Reshape([dataset.output_steps, dataset.num_output_signals]) # Shape => [batch, output_steps, num_output_signals]
])
history = compile_and_fit(model_lstm, dataset)
performance_val['LSTM'] = model_lstm.evaluate(dataset.val)
performance_test['LSTM'] = model_lstm.evaluate(dataset.test, verbose=0)

# Autoregressive Model
class AutoRegressive(tf.keras.Model):
  def __init__(self, units, num_input_signals, output_indices, output_steps):
    super().__init__()
    self.num_input_signals = num_input_signals
    self.output_indices = output_indices
    self.output_steps = output_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(self.units)
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True) # wrap LSTMCell in RNN to simplify warmup
    self.dense = tf.keras.layers.Dense(self.num_input_signals)

  def warmup(self, inputs):
    # inputs.shape => (batch, time, signals)    
    x, *state = self.lstm_rnn(inputs) # x.shape => (batch, units), state => (batch, units)  
    prediction = self.dense(x) # predictions.shape => (batch, num_input_signals)
    return prediction, state # single time-step prediction and internal state of LSTM

  def call(self, inputs, training=None):
    predictions = [] # dynamically unrolled outputs
    prediction, state = self.warmup(inputs)
    predictions.append(prediction) # insert prediction for first output step
    for n in range(1, self.output_steps): # run predictions for remaining output steps
      x = prediction  # use last prediction as input
      x, state = self.lstm_cell(x, states=state, training=training) # execute one lstm step
      prediction = self.dense(x) # convert lstm output to prediction
      predictions.append(prediction) # add prediction to output    
    predictions = tf.stack(predictions) # predictions.shape => (time, batch, signals)
    predictions = tf.transpose(predictions, [1, 0, 2]) # predictions.shape => (batch, time, signals)
    predictions = tf.gather(predictions, indices=self.output_indices, axis=-1)
    return predictions
model_autoregressive = AutoRegressive(units=16, num_input_signals=dataset.num_input_signals, output_indices=dataset.df_output_indices, output_steps=dataset.output_steps)    
history = compile_and_fit(model_autoregressive, dataset)
performance_val['AR_LSTM'] = model_autoregressive.evaluate(dataset.val)
performance_test['AR_LSTM'] = model_autoregressive.evaluate(dataset.test, verbose=0)

"""
dataset.plot_example(model_xy)
"""
"""
for example_inputs, example_outputs in dataset.train.take(1):
  print(f'Inputs shape (batch, time, signals) [features]: {example_inputs.shape}')
  print(f'Outputs shape (batch, time, signals) [labels]: {example_outputs.shape}')
print('Model Output shape:', model_xy(dataset.example[0]).shape)
"""
"""
dataset.plot_example(model_baseline)
dataset.plot_example(model_baselinerepeat)
dataset.plot_example(model_linear)
dataset.plot_example(model_linear2)
dataset.plot_example(model_dense)
dataset.plot_example(model_conv)
dataset.plot_example(model_lstm)
dataset.plot_example(model_autoregressive)
"""

# => Performance
x = np.arange(len(performance_test))
metric_name = 'mean_absolute_error'
metric_index = model_baseline.metrics_names.index('mean_absolute_error')
mae_val = [v[metric_index] / next(iter(performance_val.values()))[metric_index] for v in performance_val.values()]
mae_test = [v[metric_index] / next(iter(performance_test.values()))[metric_index] for v in performance_test.values()]
plt.ylabel('mean_absolute_error comparison')
plt.bar(x - 0.18, mae_val, 0.3, label='Validation')
plt.bar(x + 0.18, mae_test, 0.3, label='Test')
for i in range(len(x)):
  plt.text(i - 0.18, mae_val[i] + 0.01, "{0:.1f}%".format(mae_val[i] * 100), ha="center")
  plt.text(i + 0.18, mae_test[i] + 0.01, "{0:.1f}%".format(mae_test[i] * 100), ha="center")
plt.xticks(ticks=x, labels=performance_test.keys(), rotation=45)
plt.legend()
plt.show()
# print(performance_val, performance_test)
