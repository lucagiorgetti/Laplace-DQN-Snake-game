#Setting up the environment.
#The playground is a matrix having -1 on the position of the walls, 1 on the position of the snake 2 for the food and 0 for empty spaces.

using Images, ColorTypes, Plots
using Random

#game object 

mutable struct SnakeGame
    state::Matrix{Int}
    snake::Vector{CartesianIndex{2}} #head is the first tuple, queue the last
    direction::CartesianIndex{2}     #It will have to be initialized using DQN
    prev_move::CartesianIndex{2}
    score::Int
    rng::AbstractRNG
    state_size::Int
    lost::Bool

    # Custom constructor
    function SnakeGame(state_size = 10, rng = Xoshiro(42))
        state = zeros(Int, (state_size, state_size))  # Create empty grid

        # Draw walls (-1)
        state[1, :] .= -1
        state[end, :] .= -1
        state[:, 1] .= -1
        state[:, end] .= -1

        # Initialize the snake (bottom-left)
        
        snake = [CartesianIndex(state_size - 2, 2),CartesianIndex(state_size - 1, 2)]
        
        for ci in snake 
        	state[ci] = 1
        end 
        
        #initialize direction (for now)
        direction = CartesianIndex(-1,0)  #move up
        prev_move = CartesianIndex(0,0) #stay still
        
        lost = false
        
        new(state, snake, direction, prev_move, 0, rng, state_size, lost)
    end
end

#to reset the game I will simply call a new instance of the game.

# Function to plot the game state
function plot_or_update!(game::SnakeGame, plt::Union{Plots.Plot, Missing, Nothing} = missing)
    h, w = size(game.state)
    img = fill(ARGB32(1, 1, 1, 1), h, w)    # White background

    for i in 1:h, j in 1:w
        if game.state[i, j] == -1               # Walls
            img[i, j] = ARGB32(0, 0, 0, 1)  # Black
        elseif game.state[i, j] == 1            # Snake
            img[i, j] = ARGB32(0, 1, 0, 1)  # Green
        elseif game.state[i, j] == 2            # Food
            img[i, j] = ARGB32(1, 0, 0, 1)  # Red
        end
    end
    
    if ismissing(plt) || isnothing(plt)
        return plot(img, framestyle = :none)
    else   
        plot!(plt, img)
        return plt
    end 
end

# Function to place new food, not really random: the seed is fixed at the beginning of a new episode.

function sample_food!(game::SnakeGame)

    # Find empty positions
    empty_positions = findall(==(0),game.state)
    
    if isempty(empty_positions)
        return  # No space left
    end

    # Pick a random empty spot
    food_pos = rand(game.rng, empty_positions)

    # Update game state
    game.state[food_pos] = 2
end 

#update state
function update_state!(game::SnakeGame)

    game.state[game.state .== 1] .= 0 #remove previous snake positions

    # Redraw snake, this also handle removing food pixel when needed
    for ci in game.snake
        game.state[ci] = 1
    end
    
end

#function to check collision, must be modified
function check_collision(game::SnakeGame) :: Bool
    head = game.snake[1]
    return game.state[head] == -1 ||  count(==(head), game.snake) > 1 ||  game.prev_move + game.direction == CartesianIndex(0,0)   #true for a collision false otherwise, first condition check collision with the wall, second and third ones collisions with the snake itself.
end

#remove tail
function remove_tail!(game::SnakeGame)
    pop!(game.snake)
end

#function to grow maybe
function grow_maybe!(game::SnakeGame)
    new_head = game.snake[1] + game.direction
    pushfirst!(game.snake, new_head)

    # If food is eaten, place new food and increase score
    if game.state[new_head] == 2  
        sample_food!(game)
        game.score += 1
    else
        remove_tail!(game)  # Normal move (no food)
    end
end

#move wrapper, the snakes grow, if he hits the wall or himself he loses, if he does not run into food, I remove the tail 

function move_wrapper!(game::SnakeGame)
    grow_maybe!(game)  # Add new head
    
    if check_collision(game) 
        game.lost = true 
    end

    update_state!(game)
    game.prev_move = game.direction
end
                 
