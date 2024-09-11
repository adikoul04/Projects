# Top Drives Neural Network

## Overview
I am into cars, and recently I have been playing a mobile game called Top Drives. In this game, the player aquires cars which each have a rating based on certain stats about them. Some of these stats include power, weight, drivetrain, and more. I recently took the "Neural Networks and Deep Learning" course by DeepLearning.AI through Coursera. I wanted to use what I had learned in the course toward my own project, so I decided to try to build a neural network that trains based on the stats of in-game cars and predicts the ratings of cars based on stats entered by the user.

If you want to see this project in action, check it out on my [website](https://adikoul04.github.io/projects.html).

## Creating the Network
I was too lazy to create a table with all the data of every car in the game, so I found an already made spreadsheet on Google. I took my data from [here](https://docs.google.com/spreadsheets/d/1EAv9sduSWa_cbrYYWVnfICaPGGrhq6jYtr9XfJKtHs8/htmlview). Big thanks to Evan C for providing this data. **Disclamer:** This data is as of September 2019, so it is not current. Also, the RQ rating scale in this table is different than what it is currently. However, this data was still useful in creating the network. Additionally, the spreadsheet does not seem finished as all of the cars do not have data for all of the categories, so I limited my training set to 435 vehicles that had all of the data.

I used a two layer neural network for this project. The input layer was of size 11 (11 car statistics provided in the table). I arbitrarily chose a hidden layer of size 5 (I would love to know how to properly choose a hidden layer size) and the output layer of size 1 returns the calculated RQ cost. For both layers, I used a ReLU as my activation function. I could not use the sigmoid function because it outputs values between 0 and 1 and my network is not used for logistic regression. For the cost function, I used $`\sqrt{\frac{1}{m} \displaystyle\sum_{i=1}^{m} (y_i-\hat{y}_i)^2} `$ where $` m `$ is the number of training examples (435) and $` y `$ and $` \hat{y} `$ are the actual ratings and calculated ratings respectively. I trained the model with a learning rate of 2x10<sup>-7</sup> and 505 iterations (trial error to find the lowest cost), and it achieved a cost of about 7.25. After training, the model asks the user to enter inputs for a car and it uses the entered statistics and the parameters from the training to output a rating that the car would have in the game. 

## Issues and Questions
- When training the model, I had issues when calculating gradients where it said
```
divide by zero encountered in divide
dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
```
I do not know how to prevent the values of $` A2 `$ or $` 1 - A2 `$ from being 0, so I cannot avoid this error. Even though this error occurs, the training still goes on. If someone could explain this, that would be great
- I chose the size of the hidden layer arbitrarily. Also, I tweaked the learning rate and the number of iterations until I achieved a low cost. If anyone could let me know how to optimize the hidden layer size, the learning rate, and the number of iterations, that would be appreciated. Additionally, during the iterations, the cost would converge but then after a certain point it would begin to diverge, and I do not know why this is.

