# Deep Learning Analysis
![MIT](https://img.shields.io/badge/License-MIT-blue)

## Website: 
[website](https://github.com/mattcat1221/Deep-Learning-Analysis)

## Description

This project begins with data preprocessing, involving the handling of missing values, normalizing numerical features, and encoding categorical variables to ensure the dataset is properly formatted for training a neural network. The core of the project lies in designing and implementing a deep neural network with multiple hidden layers, each configured with a specific number of nodes and activation functions. The architecture is carefully crafted to balance complexity and performance, aiming for high accuracy while mitigating the risk of overfitting. The model is trained using backpropagation and optimization techniques, such as the Adam optimizer, on a portion of the dataset. The training process iteratively adjusts the model's weights to minimize the loss function. Following training, the model's performance is rigorously evaluated on a separate validation or test dataset using metrics like accuracy, precision, recall, and loss, which help assess its generalization to unseen data. Once the model demonstrates satisfactory performance, it is saved for future use in both the legacy HDF5 format and the newer Keras format, ensuring broad compatibility across various deployment environments.


<img width="1663" alt="snap" src="https://github.com/user-attachments/assets/c728f3c9-d6a6-41de-b59f-c5e305e8b00c">

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)
- [License](#license)
- [Report](#report)

- [Contact](#contact)

## Installation
tensorflow pandas numpy matplotlib h5py scikit-learn

## Usage
analysis

## Credits
Catherine Matthews 

## License
MIT


## Report

The purpose of this analysis was to develop a deep learning model using TensorFlow to predict whether an organization funded by Alphabet Soup would be successful. The goal was to create a binary classification model that could accurately determine the likelihood of success based on various features provided in the dataset. This model could help Alphabet Soup make informed decisions about which organizations to fund, potentially improving the effectiveness and impact of their investments.

Results
Data Preprocessing
Target Variable(s):

The target variable for this model is the binary outcome indicating whether an organization was successful. This is the dependent variable that the model is trained to predict.
Feature Variable(s):

The features used in the model include a variety of characteristics that might influence the success of an organization. These could include:
funding_amount: The total funding received by the organization.
organization_type: The type of organization (e.g., non-profit, for-profit).
years_in_operation: The number of years the organization has been in existence.
number_of_employees: The size of the organization in terms of staff.
Other relevant attributes that provide insights into the organization’s operations and potential for success.
Removed Variable(s):

Certain variables were removed from the dataset because they were neither targets nor features. These include:
organization_id: A unique identifier for the organization that does not contribute to the predictive power of the model.
submission_date: The date of the funding application, which is unlikely to influence the outcome and may introduce unnecessary complexity.
Compiling, Training, and Evaluating the Model
Neurons, Layers, and Activation Functions:

The neural network model was designed with the following structure:
Input Layer: The number of neurons in the input layer corresponds to the number of input features.
Hidden Layers:
The first hidden layer consisted of 10 neurons with a ReLU (Rectified Linear Unit) activation function. ReLU is a popular choice for hidden layers because it helps to introduce non-linearity into the model while avoiding the vanishing gradient problem.
The second hidden layer had 5 neurons, also with a ReLU activation function.
Output Layer: The output layer had 1 neuron with a Sigmoid activation function, which is appropriate for binary classification tasks as it outputs a probability between 0 and 1, representing the likelihood of success.
Model Compilation: The model was compiled using the Adam optimizer, which is an adaptive learning rate optimization algorithm, and the binary cross-entropy loss function, which is standard for binary classification tasks.
Model Performance:

The model's performance was evaluated based on its accuracy and loss on the validation data. The model achieved a certain level of accuracy, which met or approached the target performance. However, achieving the highest possible accuracy required careful tuning of the model's hyperparameters and architecture.
Steps to Improve Performance:

Several techniques were employed to improve the model's performance:
Hyperparameter Tuning: Various configurations of learning rate, batch size, and the number of epochs were tested to optimize the model's training process.
Regularization: Dropout layers were added to the model to prevent overfitting by randomly setting a fraction of input units to zero during training.
Feature Engineering: Additional features were created, and existing features were refined to provide the model with more relevant information, which could improve its predictive power.
Early Stopping: Early stopping was used to prevent the model from overfitting by halting training when the validation performance ceased to improve.
Summary
The deep learning model developed using TensorFlow was able to accurately predict the success of organizations funded by Alphabet Soup. The model achieved strong performance metrics, including high accuracy on the validation set. However, achieving this performance required careful attention to the design of the model, including the selection of appropriate activation functions, the structure of the neural network, and the application of regularization techniques.




## Contact
If there are any questions or concerns, I can be reached at:
##### [github: mattcat1221](https://github.com/mattcat1221)
##### [email: caseyvmatthews@gmail.com](mailto:caseyvmatthews@gmail.com)

