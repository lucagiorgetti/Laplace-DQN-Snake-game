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

## Improvements Applied to DQN for SnakeGame

1. **Two frames as input state**  
   Using two consecutive frames instead of one allows the Q-network to observe transitions, which helps learning motion dynamics.

2. **Reducing the number of actions per state**  
   The snake cannot move in the opposite direction to its current movement without losing.  
   Hence, we reduce the possible actions from 4 to 3 per state.

3. **Modified DQN Loss**  
   The standard DQN loss is:
   <div align="center">
  <img src="https://render.githubusercontent.com/render/math?math=L_i(\theta_i)%20%3D%20\mathbb{E}_{(s,a,r,s')\sim U(D)}%5B(r%20%2B%20\gamma%20\max_{a'}%20Q(s',a';\theta_i^-) - Q(s,a;\theta_i))^2%5D" alt="DQN Loss">
</div>

   However, the maximum over all actions can lead to overestimation of Q-values, which impairs performance.  
   By **excluding losing actions** from the maximum selection, we mitigate this effect.

After these improvements the snake reaches a score of 33. The best game played is the following:

![Improved DQN, best game](trainer_gifs/very_long_double_training3.gif)


