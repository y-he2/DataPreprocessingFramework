# Efficient Python Data & Signal Processing Framework

## Main Features
* This is a framework which Im progressively improving, during the use as a lab enviroment for my brainstorming experiments. 
* Its based on Pandas, Dask/Ray, Numpy, Scipy, Keras, Tensorflow/Pytorch etc. 
* It could enable efficient data and signal processing, feature engineering and machine learning. 
* Main features including:
  * mp_utility: A Multiprocessing based shared memory 3D tensor apply function. Can be used to apply any function over the last 2 axis, parallelized over the first axis. 
  * tensor_utility: some useful function for array manipulation, including generate signature matrix for sequence of matrices. 
  * signal_utility: Contains various useful functions for processing signals stored in Numpy/Scipy/Pandas.
  * preprocessor.py: some Pandas helper. 
I uploaded them in a rush so theres currently no explaination nor documentation. 
Currently the lab enviroment is not live on Github so I may rip this repo and recreate a proper one. 

## Example Experiment (MSCRED) 
The files poc_signature_anomaly_detection.ipynb + multi_ts_pearson_func.py + signature_matrix_func.py include an attempted experiment on doing anomaly detection on a set of Forex closing price data timeseries, using the environment of this framework. The MSCRED model was a reimplementation of the paper:

https://arxiv.org/abs/1811.08055

Instead of compute and store the generated tensor data, it was recomputed on each run using the parallel tensor apply feature. 
* signature_matrix_func.py: An parallel function to compute the "signature matrix" from a sequence of matrices, which is basically taking the inner product of pair of rows (timeseries) for each one in the sequence. 
* multi_ts_pearson_func.py: Basically the same but using Pearsons Correlation with some modification to take in account of diff changes in timeseries. 

Unfortunately this experiment didnt gave any significant result, but it should still give an idea of the necessary work to be carried out in order to do this kind of anomaly detection. The cause could be many, one of which I noticed was the training error of the MSCRED on generated data converges (never stops) to zero, which is an indication of information leaking in the traning data. 
