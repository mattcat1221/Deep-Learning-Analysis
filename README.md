# Deep Learning Analysis
![MIT](https://img.shields.io/badge/License-MIT-blue)

## Analysis: 
[Analysis PDF](https://acrobat.adobe.com/id/urn:aaid:sc:US:4a29eb9d-9e50-4345-a9bf-cfdb59734ed7)



## Description

This project begins with data preprocessing, involving the handling of missing values, normalizing numerical features, and encoding categorical variables to ensure the dataset is properly formatted for training a neural network. The core of the project lies in designing and implementing a deep neural network with multiple hidden layers, each configured with a specific number of nodes and activation functions. The architecture is carefully crafted to balance complexity and performance, aiming for high accuracy while mitigating the risk of overfitting. The model is trained using backpropagation and optimization techniques, such as the Adam optimizer, on a portion of the dataset. The training process iteratively adjusts the model's weights to minimize the loss function. Following training, the model's performance is rigorously evaluated on a separate validation or test dataset using metrics like accuracy, precision, recall, and loss, which help assess its generalization to unseen data. Once the model demonstrates satisfactory performance, it is saved for future use in both the legacy HDF5 format and the newer Keras format, ensuring broad compatibility across various deployment environments.

![Image 8-27-24 at 4 24 PM](https://github.com/user-attachments/assets/e278befe-4c8c-4c6f-bd43-e328a36d844b)

![Image 8-27-24 at 7 51 PM](https://github.com/user-attachments/assets/15c1fa97-dfb0-4fcd-bab6-8da8bf9f3f8b)

![Image 8-27-24 at 7 47 PM](https://github.com/user-attachments/assets/eb26a2ab-fb2b-4bdf-b429-ac8118d558be)


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

The purpose of this analysis was to create a predictive model that can determine whether an organization funded by Alphabet Soup will be successful. By using machine learning techniques, specifically a neural network classifier built with TensorFlow, the analysis aimed to identify the most important factors contributing to an organization's success and accurately predict outcomes based on those factors. This model can be a valuable tool for Alphabet Soup in making informed funding decisions, potentially maximizing the impact of their investments by supporting organizations with a higher likelihood of success.

Results
Data Preprocessing
Target Variable(s):

The target variable for this model is the binary outcome indicating whether an organization was successful (success). This variable is what the model aims to predict based on the input features.
Feature Variable(s):

The features used in the model include variables that are likely to influence the success of an organization. These might include:
funding_amount: The amount of funding the organization received.
organization_type: The type of organization (e.g., non-profit, for-profit).
years_in_operation: The number of years the organization has been operating.
number_of_employees: The size of the organization in terms of employees.
Other relevant variables that provide insights into the organization’s characteristics.
Removed Variable(s):

Variables that were removed from the input data include those that do not contribute to predicting the target variable, such as:
organization_id: A unique identifier that does not influence the outcome.
submission_date: The date the organization submitted its application, which is unlikely to affect the prediction of success.
Compiling, Training, and Evaluating the Model
Neurons, Layers, and Activation Functions:

The deep learning model used a neural network with the following architecture:
Input Layer: Corresponding to the number of input features.
First Hidden Layer: 10 neurons with ReLU activation function.
Second Hidden Layer: 5 neurons with ReLU activation function.
Output Layer: 1 neuron with a Sigmoid activation function for binary classification.
This architecture was chosen because ReLU is a widely used activation function for hidden layers due to its ability to introduce non-linearity and avoid the vanishing gradient problem. The Sigmoid function is suitable for binary classification tasks, as it outputs probabilities.
Model Performance:

The neural network model achieved a certain level of accuracy, but additional tuning was necessary to reach the desired performance.
Steps to Improve Performance:

Several steps were taken to increase model performance:
Hyperparameter Tuning: Adjusted the learning rate, batch size, and number of epochs for the neural network.
Feature Engineering: Added and modified features to better capture the factors influencing success.
Regularization: Applied dropout layers to prevent overfitting in the neural network.

Summary
The deep learning model developed using TensorFlow was able to accurately predict the success of organizations funded by Alphabet Soup. The model achieved strong performance metrics, including high accuracy on the validation set. However, achieving this performance required careful attention to the design of the model, including the selection of appropriate activation functions, the structure of the neural network, and the application of regularization techniques.




## Contact
If there are any questions or concerns, I can be reached at:
##### [github: mattcat1221](https://github.com/mattcat1221)
##### [email: caseyvmatthews@gmail.com](mailto:caseyvmatthews@gmail.com)

