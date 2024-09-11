# Car Driving RL Model

## Overview
I am interested in AI/ML, and recently I watched a video where a Youtuber created an AI to play the car video game _Trackmania_ through reinforcement learning. I thought this was really cool and it inspired me to pursue a similar project (although much simpler), so I developed an RL model that learns to drive a car in a basic game I created using pygame.

If you want to see this project in action, check it out on my [website](https://adikoul04.github.io/projects.html).

## Creating the Game
Before creating a learning model, I first needed an environment which the model could learn in. I created a simple game which spawns a car on a circular track. The user can use the arrow keys to drive the car and try to achieve the fastest lap time possible. I did not put too much effort into making the game "good" in any way; my primary focus was building the model and I simply needed an environment which includes a car that could drive. 

**Note:** If you want to play the game yourself, run "game.py" and use the arrows to try to beat my best lap time of 4.93 seconds

## Creating the Model
I created the model using an Epsilon-Greedy Q-Learning algorithm. For the state of the car, I included the car's speed, angle around the track, and angle of the car relative to the direction of the track at the car's position. I did not include the car's position when defining a state because the track is circular so the posiiton of the car does not really matter. However, I might try including the car's position in further training to see if that produces a faster lap time. The actions of the car are each of the four arrow keys (accelerate, brake, turn left, turn right) and neutral, which is not pressing any of the arrows. 

In the model that completed a full lap, the reward is calculated by taking the sqrt of the product of the car's speed and its angle around the track if the car is on the track. If it is off the track, the reward is a negative constant. After about 60,000 iterations, the car was able to complete a lap around the track in 8.24 seconds. However, the behavior of the car is slower than optimal and I was able to manually set a lap time much faster than the model, so I am testing different rewards to try to achieve a faster lap time

## Issues and Questions
As stated previously, even though the model was able to complete a full lap, it was slower than I could manually drive, so I am trying to change the reward calculation to incentivize the model to drive faster. If anyone has ideas on how to make the car drive faster, please let me know. 

