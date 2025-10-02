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
   
   ![DQN Loss](https://latex.codecogs.com/svg.latex?L_{i}(\theta_{i})%3D\mathbb{E}_{(s,a,r,s^{\prime})\sim\mathrm{U}(D)}\left[\left(r%2B\gamma\operatorname*{max}_{a^{\prime}}Q(s^{\prime},a^{\prime};\theta_{i}^{-})-Q(s,a;\theta_{i})\right)^{2}\right]).
   
   However, the maximum over all actions can lead to overestimation of Q-values, which impairs performance.  
   By **excluding losing actions** from the maximum selection, we mitigate this effect.

After these improvements the snake reaches a score of 33. The best game played is the following:

![Improved DQN, best game](trainer_gifs/very_long_double_training3.gif)

(See also this [DragonWarrior15](https://github.com/DragonWarrior15/snake-rl) from which I took inspiration).
The following table shows a list of the hyperparameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Discount factor ($\gamma$) | 0.97 | Balances the importance of immediate versus future rewards. |
| Learning rate ($\eta$) | 0.0005 | Step size used by the optimizer (RMSProp). |
| Momentum ($\rho$) | 0.9 | Momentum parameter used by the optimizer. |
| Batch size | 64 | Number of experiences sampled from the buffer per training step. |
| Target update rate | 1000 | Frequency (in number of mini-batches) of synchronizing the target network with the Q-network. |
| Replay buffer capacity | 50,000 | Number of experiences stored in the buffer; training starts once the buffer is full. |
| Initial exploration rate ($\epsilon$) | 1.0 | Probability of choosing a random action at the beginning of training. |
| Final exploration rate | 0.05 | Minimum exploration rate during training. |
| Exploration decay rate | $1 \times 10^{-6}$ | Amount subtracted from $\epsilon$ after each mini-batch. |
| n\_batches | 1,200,000 | Total number of mini-batch updates performed. |
| Eating reward | 1.0 | Reward the agent gets for eating an apple. |
| Suicide penalty | -1.0 | Penalty the agent gets if it collides with a wall or itself. |
| Male di vivere | -0.01 | Penalty the agent receives for making a step without dying or eating an apple. |




