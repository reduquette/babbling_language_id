import numpy as np
from keras import models, layers, losses
import pandas as pd
import matplotlib.pyplot as plt

data_english = np.load('data_english.npy')
data_french = np.load('data_french.npy')

#Prepare data to pass to model
labels = np.asarray([1] * len(data_english) + [0] * len(data_french))
data = np.concatenate((data_english, data_french))
print(labels.shape)
print(data.shape)

# shuffle array order

shuffler = np.random.permutation(len(data))
labels_shuffled = labels[shuffler]
data_shuffled = data[shuffler]

data_shuffled = data_shuffled.reshape(-1, 128, 157, 1)

#create CNN
#link to source for architecture: https://docs.google.com/document/d/1ydh6-a05urM-wbKd0n8rrnw7aTUooVMV7kAA1d9tIOY/edit?usp=sharing

# (7×7, 16),(5×5, 32),(3×3, 64),(3×3, 128),(3×3, 256)
#CNN architecture -- same as above
model2 = models.Sequential()
model2.add(
    layers.Conv2D(16, (7, 7), activation='relu',
                  input_shape=(128, 157,
                               1)))  #input shape will need to be changed
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(32, (5, 5), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(256, (3, 3), activation='relu'))

#adding BLSTM
model2.add(layers.Flatten())  #flatten CNN output
model2.add(layers.Reshape((3840, 1)))
model2.add(layers.Bidirectional(layers.LSTM(256), merge_mode='concat'))
model2.add(layers.Dense(2, activation='softmax'))

model2.summary()

model2.compile(
    optimizer='adam',
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model2.fit(
    data_shuffled,
    labels_shuffled,
    batch_size=100,
    epochs=10,
    verbose=1,
    validation_split=.3)
try:
    hist_df = pandas.DataFrame(history.history)
    hist_csv_file = 'history_model2.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
except:
    print('failed to save model as pandas csv')

model2.save("model2_all")

#plot history
#plot accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model2_accuracy.png')
# plot loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model2_loss.png')
