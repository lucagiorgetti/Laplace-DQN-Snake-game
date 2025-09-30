# Laplace-DQN-Snake-game
This repository contains my master's thesis project. It consists in two parts:
1. The application of the Deep-Q-learning (DQN) algorithm to the classic Snake game.
2. An attempt to extend the algorithm using the Laplace approximation, which will be explained in the next sections.
   
The snake is allowed to move inside a 2-dimensional playing field (game map) surrounded by walls. At each discrete interval (a time step), the snake must move forward, turn left, or turn right as the game requires that the snake cannot stop moving. 
The game will generate and place one piece of food on the game map whenever there is no food left. When the snake moves onto a piece of food, the food is eaten and the snake’s length grows by one.
The goal is to eat as many pieces of food as possible without ending the game by colliding the snake into itself or the walls.
The snake starts the game, as a single point, from the bottom left of the grid.

## DQN
A vanilla DQN, with a single frame as state of the Q-nework leads to poor results (9 eaten apples). The following gif shows the best game after 100 000 mini-batches of training:

![Vanilla DQN, best game](trainer_gifs/very_long_training1.gif)

Applying a few improvements:
1. Two frames instead of one as a state: it is beneficial for the Q-network to see transitions of frames.
2. Reducing the number of possible actions in each state to 3 instead of 4. Indeed, the action prescribing to the snake to move in the opposite direction to the one it is already moving is always loosing. 
3. In the loss of DQN: $$ L_{i}(\theta_{i})=\mathbb{E}_{(s,a,r,s^{\prime})\sim\mathrm{U}(D)}\left[\left(r+\gamma\operatorname*{max}_{a^{\prime}}Q(s^{\prime},a^{\prime};\theta_{i}^{-})-Q(s,a;\theta_{i})\right)^{2}\right]$$, the maximum leads to an overstimation of Q-values and a subsequent impairment of the performance. Excluding the loosing actions from the selection of the maximum mitigates this effect.

After these improvements the snake reaches a score of 33. The best game played is the following:

![Improved DQN, best game](trainer_gifs/very_long_double_training3.gif)


