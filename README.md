# Laplace-DQN-Snake-game

In the game, the snake is allowed to move inside a 2-dimensional playing field (game map) surrounded by walls. At each discrete interval (a time step), the snake must move forward, turn left, or turn right as the game requires that the snake cannot stop moving. 
The game will randomly (based on a chosen distribution) generate and place one piece of food on the game map whenever there is no food on the map. When the snake moves onto a piece of food, the food is eaten and the snake’s length grows by one.
The goal is to eat as many pieces of food as possible without ending the game by colliding the snake into itself or the walls.
The snake starts the game, as a single point, from the bottom left of the grid.

The training algorithm integrates the Laplace approximation with DQN and works as follows:
1. Initialize the parameter for the Laplace approximation: $\gamma^2$ for the Gaussian prior and $\beta$ for the Gaussian likelihood.
2. Experience with $\epsilon$-greedy policy untill therere are sufficient samples in the replay buffer.
3. Pre-Train the neural network
4. Compute the Hessian using the current $\gamma^2$ and $\beta$, do model selection to update them, recompute Hessian, train again, go on until convergence of $\gamma^2$ and $\beta$.
5. Apply Laplace Approximation. From now on I can use the confidence level of the prediction to guide exploration towards more uncertain actions.
6. Restart to acquire experience and refill the buffer.
7. Continue.
