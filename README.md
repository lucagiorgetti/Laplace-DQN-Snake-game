# Laplace-DQN-Snake-game
These repository contains my master's thesis project. It consists in two parts:
1. The application of the Deep-Q-learning (DQN) algorithm to the classic Snake game.
2. An attempt to extend the algorithm using the Laplace approximation, which will be explained in the next sections.
   
The snake is allowed to move inside a 2-dimensional playing field (game map) surrounded by walls. At each discrete interval (a time step), the snake must move forward, turn left, or turn right as the game requires that the snake cannot stop moving. 
The game will generate and place one piece of food on the game map whenever there is no food left. When the snake moves onto a piece of food, the food is eaten and the snake’s length grows by one.
The goal is to eat as many pieces of food as possible without ending the game by colliding the snake into itself or the walls.
The snake starts the game, as a single point, from the bottom left of the grid.

## DQN
A vanilla DQN, with a single frame as state of the Q-nework leads to poor results. The following gif shows the best game after 100_000 mini-batches of training:

