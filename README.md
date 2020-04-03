# lstm-qrs-detector
## CNN-LSTM based QRS detector for ECG signals

This project implements a deep learning based QRS detector for ECG signals. Specifically, a hybrid CNN-LSTM model is used. On the test set, this model achieves an f1 of 0.88 and accuracy of 0.97. To get right to the punchline, here's the model:
```
#first CNN
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=7, padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1,2)))
model.add(Dropout(0.1))

#second CNN
model.add(Conv2D(filters=32, kernel_size=5, padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1,2)))
model.add(Dropout(0.1))

#third CNN
model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1,5)))
model.add(Dropout(0.1))

#first LSTM. note that we need to do a timedistributed flatten as a transition from CNN to LSTM
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(units=75, return_sequences=True)))
model.add(Dropout(0.1))
model.add(BatchNormalization())

#second LSTM
model.add(Bidirectional(LSTM(units=25, return_sequences=True)))
model.add(Dropout(0.1))
model.add(BatchNormalization())

#dense layer
model.add(TimeDistributed(Dense(5, activation='relu')))
model.add(Dropout(0.1))
model.add(BatchNormalization())

#activation layer
model.add(TimeDistributed(Dense(2, activation='softmax')))
```

To run the code in this project, run the following notebooks:
1. ```pull_qt_db.ipynb```: This notebook pulls data from the Physionet QT database, which is the data source for this project
2. ```preprocess.ipynb```: This notebook applies some filtering, baseline wander removal, and calculates the scalogram (ie continuous wavelet transform)
3. ```gen_train_test_data.ipynb```: This notebook partitions the data into training and testing sets
4. ```qrs_detector.ipynb```: This notebook trains the model and evaluates its performance

The remainder of this readme will cover the different steps in the analysis pipeline.

## 1. Download/Parse the Data
The wfdb library is used to download data from the Physionet QT database. Small sections of each file are labeled with P,Q,R,S, and T points. 

An example plot of the ECG data, along with QRS labels:
![sel100](https://github.com/nerajbobra/lstm-qrs-detector/blob/master/plots/sel100.png "sel100")

## 2. Preprocess the Data
First, the baseline wander is removed. Instead of using an FIR filter, which will inevitabely remove frequencies of interest regardless of how well it is designed, the method of local linear regression is used instead. The idea is basically to calculate a linear regression over a window of about 1.5 seconds, and then define the "baseline" to be the center of that window. Then shift the window forward by one point, and repeat. The process is extremely efficient because the linear regression can be solved in a closed form analytical solution, as explained below:

<img src="https://github.com/nerajbobra/lstm-qrs-detector/blob/master/linear_regression.jpg" width="600">

An example result:
![Baseline Wander Removal](https://github.com/nerajbobra/lstm-qrs-detector/blob/master/baseline_filtered/sel31_ch1.png "Baseline Wander Removal")

Next, the scalogram (continuous wavelet transform) is calculated. Since there isn't a lot of energy above 60Hz, the signal is first downsampled to 125Hz using an anti-aliasing lowpass filter. The wavelet transform is then calculated using the morlet wavelet. An example result:
![Scalogram](https://github.com/nerajbobra/lstm-qrs-detector/blob/master/cwt/sele0170_ch1.png
 "Scalogram")

## 3. Train the Model
For training, a validation split of 10% was used and an early stopping criterion was implemented based on the validation loss. 
The accuracy and loss over the training session:

<img src="https://github.com/nerajbobra/lstm-qrs-detector/blob/master/plots/accuracy.png" width="775">
<img src="https://github.com/nerajbobra/lstm-qrs-detector/blob/master/plots/loss.png" width="775">

Additionally, the ROC:
![ROC](https://github.com/nerajbobra/lstm-qrs-detector/blob/master/plots/ROC.png "ROC")

## 4. Evaluate the Model
On the testing set, f1=0.88 and accuracy=0.97. An example classification result:
![Prediction](https://github.com/nerajbobra/lstm-qrs-detector/blob/master/predictions/220.png "Prediction")
 
## Other Notes
The data used for this analysis is available at the following link: 
https://physionet.org/content/qtdb/1.0.0/

The following matlab tutorial was used as a reference: 
https://www.mathworks.com/help/signal/examples/classify-ecg-signals-using-long-short-term-memory-networks.html
