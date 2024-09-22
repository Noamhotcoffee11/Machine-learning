import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import pandas as pd
import matplotlib.pyplot as plt

TF_ENABLE_ONEDNN_OPTS=0

# Load MNIST data
(x_train, l_train), (x_test, l_test) = mnist.load_data()

for i in range(4):
    plt.figure()
    plt.imshow(x_train[i], 'gray')

# One-hot encode labels
y_train = np.zeros((l_train.shape[0], l_train.max() + 1), dtype=np.float32)
y_train[np.arange(l_train.shape[0]), l_train] = 1

y_test = np.zeros((l_test.shape[0], l_test.max() + 1), dtype=np.float32)
y_test[np.arange(l_test.shape[0]), l_test] = 1

# Define the model function
def Model(Neurons, activation, Learning_rate, batch_size, epochs):
    model_1 = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(Neurons, activation=activation),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model with the passed learning rate
    model_1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=Learning_rate),
                    loss='mean_squared_error',
                    metrics=['accuracy'])
    
    # Train the model and return history
    history = model_1.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model_1, history

# Initialize an empty DataFrame to store all results
results_df = pd.DataFrame()

# Define parameter lists
neurons_list = [15, 30, 100]
activations = ['sigmoid']
learning_rates = [0.1, 3.0]
batch_size = 16
epoch = [10, 30]  

# Train each model and store the results
for Neurons in neurons_list:
    for activation in activations:
        for Learning_rate in learning_rates:
            for epochs in epoch:
                # Train the model with the current configuration
                model_1, history = Model(Neurons, activation, Learning_rate, batch_size, epochs)
                
                # Evaluate the model on the test data
                test_loss, test_acc = model_1.evaluate(x_test, y_test, verbose=0)
                
                # Get training accuracy and loss from history
                train_loss = history.history['loss'][-1]
                train_acc = history.history['accuracy'][-1]
                
                # Convert the history to a DataFrame
                history_df = pd.DataFrame(history.history)
                
                # Add columns for the current model's hyperparameters and results
                history_df['activation'] = activation
                history_df['Neurons'] = Neurons
                history_df['L_rate'] = Learning_rate
                history_df['Epochs'] = epochs
                history_df['Batch_size'] = batch_size
                history_df['test_accuracy'] = int(100 * test_acc)
                history_df['test_accuracy'] = history_df['test_accuracy'].astype(str) + '%'
                history_df['test_loss'] = int(test_loss * 100)
                history_df['test_loss'] = history_df['test_loss'].astype(str) + '%'
                
                # Format accuracy and loss
                history_df['accuracy'] = round((history_df['accuracy'] * 100))
                history_df['accuracy'] = history_df['accuracy'].astype(str) + '%'
                history_df['loss'] = round((history_df['loss'] * 100))
                history_df['loss'] = history_df['loss'].astype(str) + '%'
                
                # Reorder columns
                columns = list(history_df.columns)
                columns.remove('accuracy')
                columns.remove('loss')
                columns.extend(['accuracy', 'loss'])
                history_df = history_df.reindex(columns, axis=1)
                
                # Get last epoch data
                last_epoch = history_df.iloc[-1]
                last_epoch = pd.DataFrame(last_epoch).T
                
                # Concatenate results
                results_df = pd.concat([results_df, last_epoch], ignore_index=True)

# Save the cumulative results to a CSV file
results_df.to_csv('training_results.csv', index=False)

print(results_df)
