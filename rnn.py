import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from collections import deque
import random
from sklearn import preprocessing
import keras
import sys
from keras.models import Sequential
from keras.layers import LSTM, Dense


SEQ_LEN = 120
BATCH_SIZE = 64
EPOCHS = 3

def preparePreprocess_df():
    seqData = []
    def preprocess_df(df):
        prevMeasures = deque(maxlen=SEQ_LEN)
        for i in df.values:
            prevMeasures.append(i[0])
            if len(prevMeasures) == SEQ_LEN:
                seqData.append([np.array(prevMeasures), i[-1]])

    def outputXY():
        random.shuffle(seqData)

        X = []
        y = []

        for seq, target in seqData:
            X.append(seq)
            y.append(target)

        X = np.array(X)
        X = X.reshape(*X.shape,1)
        y = np.array(y)
        return X, y

    return preprocess_df, outputXY


if __name__ == "__main__":

    t = np.linspace(0,100,10000)

    if len(sys.argv) > 1:
        model = keras.models.load_model(sys.argv[1])
    else:
        validationPreprocess_df, validationXY = preparePreprocess_df()
        trainingPreprocess_df, trainingXY     = preparePreprocess_df()
        for tag in range(1,5):
            main_df = pd.DataFrame()
            main_df["time"] = t
            main_df.set_index("time", inplace=True)
            main_df[f"sin{tag}"] = np.sin(tag*t)
            main_df[f"target{tag}"] = tag

            last_5pct = t[-int(0.05*len(t))]
            validation_main_df = main_df[(main_df.index >= last_5pct)]
            main_df = main_df[(main_df.index < last_5pct)]

            validationPreprocess_df(validation_main_df)
            trainingPreprocess_df(main_df)


            # preprocess_df(main_df)

        X_train, y_train = trainingXY()
        X_val, y_val = validationXY()


        print(X_train.shape)
        model = Sequential()
        model.add(LSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
        model.add(Dense(32,activation='relu'))
        model.add(Dense(1,activation='relu'))

        opt = keras.optimizers.Adam(lr=0.01, decay=1e-6)

        model.compile(
                loss='mean_squared_error',
                optimizer=opt,
                metrics=['mean_squared_error']
                )

        history = model.fit(
                X_train, y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(X_val, y_val),
                )


        score = model.evaluate(X_val, y_val, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save("models/modelolegal")

    window = deque(maxlen=2000)
    for time in t:
        window.append(np.sin(time*10))
        x = np.array(window)
        x = x.reshape(1,*x.shape,1)
        print(model.predict(x)[-1][-1])
