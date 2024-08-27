# Deep Learning Analysis
![MIT](https://img.shields.io/badge/License-MIT-blue)

## Website: 
[website](https://github.com/mattcat1221/Deep-Learning-Analysis)

## Description

This project begins with data preprocessing, involving the handling of missing values, normalizing numerical features, and encoding categorical variables to ensure the dataset is properly formatted for training a neural network. The core of the project lies in designing and implementing a deep neural network with multiple hidden layers, each configured with a specific number of nodes and activation functions. The architecture is carefully crafted to balance complexity and performance, aiming for high accuracy while mitigating the risk of overfitting. The model is trained using backpropagation and optimization techniques, such as the Adam optimizer, on a portion of the dataset. The training process iteratively adjusts the model's weights to minimize the loss function. Following training, the model's performance is rigorously evaluated on a separate validation or test dataset using metrics like accuracy, precision, recall, and loss, which help assess its generalization to unseen data. Once the model demonstrates satisfactory performance, it is saved for future use in both the legacy HDF5 format and the newer Keras format, ensuring broad compatibility across various deployment environments.

![Image 8-27-24 at 4 24 PM](https://github.com/user-attachments/assets/e278befe-4c8c-4c6f-bd43-e328a36d844b)
![Image 8-27-24 at 7 51 PM](https://github.com/user-attachments/assets/15c1fa97-dfb0-4fcd-bab6-8da8bf9f3f8b)
![Image 8-27-24 at 7 51 PM](https://github.com/user-attachments/assets/fb53d2ab-6040-4556-aa40-5f08a3e66b94)
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
The neural network model proved to be an effective choice for predicting the success of organizations funded by Alphabet Soup. The model achieved a good level of accuracy and provided clear insights into the most influential factors driving organizational success. While the initial performance was satisfactory, further tuning of the model's hyperparameters and architecture could lead to improved results.

To solve the problem of predicting the success of organizations funded by Alphabet Soup, a different model that could be effectively used is a Random Forest classifier. Here’s why and how this model could be applied:

Why Use a Random Forest Classifier?
Robustness Against Overfitting:

Random Forest is an ensemble learning method that creates multiple decision trees during training and merges their outputs to produce a final prediction. This approach reduces the risk of overfitting, which can be a concern in deep learning models, especially when the dataset is not very large. Each tree in the forest is trained on a random subset of the data, which helps to smooth out the variance and avoid overfitting.
Handling Non-Linear Relationships:

Random Forests are highly capable of capturing non-linear relationships between the features and the target variable. In scenarios where the relationship between inputs and outputs is complex and non-linear, Random Forests can be very effective because they do not require any assumptions about the underlying data distribution.
Feature Importance:

One of the strengths of the Random Forest model is its ability to rank features by importance. This feature is beneficial when working with datasets with many variables, as it can help identify the most influential features that contribute to the prediction of success. This insight can be valuable for refining the model and understanding the factors that drive success.
Interpretability:

Compared to neural networks, Random Forests are more interpretable. While neural networks are often considered "black boxes" due to their complexity and difficulty in interpreting individual neurons' contributions, Random Forests provide more transparent decision-making paths, making it easier to explain predictions to stakeholders.
Versatility:

Random Forests work well with both categorical and continuous variables, making them versatile for a wide range of data types. Additionally, they are less sensitive to outliers and missing data, which can be beneficial if the dataset contains noise or incomplete records.


Recommendation: For future analyses, it is recommended to explore more advanced neural network architectures or ensemble methods such as Gradient Boosting Machines for classification tasks like this. These models are particularly effective when dealing with datasets that contain a mix of categorical and numerical features. Additionally, further hyperparameter tuning and experimentation with different layers could potentially enhance the model's performance.



## Contact
If there are any questions or concerns, I can be reached at:
##### [github: mattcat1221](https://github.com/mattcat1221)
##### [email: caseyvmatthews@gmail.com](mailto:caseyvmatthews@gmail.com)

