# Door-Open-Closed
The program detects if a door is open or closed. 

Environment requirements: numpy, opencv, scikit-learn

For each state of the door (open/closed), 4 videos (each having 1min duration, 30fps) were recorded.
Half of them were used for training and half of them were used for testing.

Each frame was processed and liniarized in order to obtain the features of the image, which were used in training.

Several classifiers were tested yielding the following results.
  Nearest Neighbors:0.6033898305084746
  Linear SVM:0.9169491525423729
  RBF SVM:1.0
  Gaussian Process:0.0
  Decision Tree:0.8677966101694915
  Random Forest:0.43050847457627117
  Neural Net:0.8169491525423729
  AdaBoost:0.7576271186440678
  Naive Bayes:0.31186440677966104
  QDA:0.8322033898305085

