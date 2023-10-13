# Neural-Network-Implementation

Neural Network Implementation and Evaluation 

Overview:

This repository contains an implementation of a neural network from scratch using Python and NumPy. The neural network is designed with one input layer, one hidden layer, and one output layer. The code supports flexible configurations, allowing users to input parameters such as the learning rate, the number of neurons in the input layer, and the number of hidden layer neurons. The implementation is evaluated using the Boston Housing dataset through 5-fold and 10-fold cross-validation.


Contents:

1. Code/Model: Contains the main implementation of the neural network, including functions for forward propagation, backpropagation, training, and cross-validation. The code is modular and well-commented for easy understanding.

2. housing.csv: The dataset used for training and evaluation. It contains features such as RM, LSTAT, and PTRATIO, with the target variable being MEDV.

3. README.md: Provides essential information about the repository, including setup instructions, usage guidelines, and results interpretation.

4. Visualization: Gives visual representation through plots of 5-fold and 10-fold data.


Dependencies:

Python 3.x
NumPy: pip install numpy
Pandas (for data handling): pip install pandas
scikit-learn (for KFold cross-validation): pip install scikit-learn


Download the dataset from the provided link and save it as housing.csv in the repository directory.


Training and Evaluation:
Open final-model-10fold,py, and set the desired parameters (learning rate, hidden layer neurons, etc.).
Run the script to train the neural network and perform 5-fold and 10-fold cross-validation on the Boston Housing dataset.
Results:

The script will output the mean absolute error for different configurations of hidden neurons and learning rates, both for 5-fold and 10-fold cross-validation. Interpret the results to understand the model's performance under various settings.

Issues and Improvements:

The code currently faces issues related to numerical stability, leading to NaN values during evaluation. Addressing these issues, such as using more stable activation functions, can improve the reliability of the results.
Contributing:

Contributions and improvements to the code are welcome! Fork the repository, make your changes, and create a pull request with a clear description of the modifications.

Acknowledgments:

The implementation is based on fundamental concepts of neural networks and backpropagation. Thanks to the open-source community for valuable resources and tutorials.
Feel free to explore, experiment, and contribute to enhancing this neural network implementation!
