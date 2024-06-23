import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

dataset = pd.read_json("C:\\Users\\furka\\Desktop\\training_data2.json")

dataset.replace('', np.nan, inplace=True) 
dataset.dropna(inplace=True)  

X = dataset.drop(columns=["prognosis"])
y = dataset["prognosis"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("Size:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.expand_dims(X_train.values, axis=2).astype('float32')
X_test = np.expand_dims(X_test.values, axis=2).astype('float32')

model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

model.save("hastalık_tahmin_modeli.h5")
