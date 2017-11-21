# ConvLSTM
Implementation of a Convolutional LSTM with Keras for video segmentation.

Data is provided in folder data as a set of videos (mp4 format) and corresponding segmentation mask (extension _label) in the filename.

Run file 

```
	lstm_train_fcn.py 
```

to train train the network. 
Training should take 1 hour per video sequence of 1000 frames in an Nvidia TitanX	