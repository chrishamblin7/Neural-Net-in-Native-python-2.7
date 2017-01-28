# Neural-Net-in-Native-python-2.7
A basic neural Net written in native python 2.7 using batch gradient descent back propagation algorithm 
Requires no fancy depencies, just matplotlib, numpy, and scikit learn to generate some of the data sets. Comes with a basic
contour plotter to show net predictions as gradient descent tunes weights.
Currently comes with 4 possible data sets and accompanying labels, (Xmoon, ymoon), (Xcirc, ycirc), (Xblob, yblob), and (Xcube, ycube), where X is a matrix of the the coordinates for each data point and y is a vector of the label for each data point. Change the number of sample data points at the top of the script. Net parameters are listed as arguments in the build_model function, and can be adjusted in the last line of the script when build_model is called. As this model currently employs batch gradient descent (not stochastic grad desc), it will sometimes get stuck in local minimums, especially when trying to predict blob data, which has three labels instead of 2. Enjoy exploring!
