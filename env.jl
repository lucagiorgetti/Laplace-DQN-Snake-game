#Setting up the environment.
#The playground is a matrix having -1 on the position of the walls, 1 on the position of the snake (-7, 7 respectively for the queue and the head), 2 for the food and 0 for empty spaces.

using Images, ColorTypes, Plots
using Random

mutable struct SnakeGame
    state::Matrix{Int}
    score::Int
    rng::AbstractRNG
    state_size::Int
    prev_food::Tuple{Int, Int}

    # Custom constructor
    function SnakeGame(state_size = 10, rng = Xoshiro(42))
        state = zeros(Int, (state_size, state_size))  # Create empty grid

        # Draw walls (-1)
        state[1, :] .= -1
        state[end, :] .= -1
        state[:, 1] .= -1
        state[:, end] .= -1

        # Initialize the snake (bottom-left), -7 is the queue, 7 is the head
        state[end - 2, 2] = 7
        state[end - 1, 2] = -7

        # Place the first food
        prev_food = (5,5)
        state[5,5] = 2  # Place food

        new(state, 0, rng, state_size, prev_food)
    end
end

#to reset the game I will simply call a new instance of the game.

# Function to plot the game state
function plot_state(game::SnakeGame)
    h, w = size(game.state)
    img = fill(ARGB32(1, 1, 1, 1), h, w)    # White background

    for i in 1:h, j in 1:w
        if game.state[i, j] == -1               # Walls
            img[i, j] = ARGB32(0, 0, 0, 1)  # Black
        elseif game.state[i, j] in [-7, 1, 7]        # Snake
            img[i, j] = ARGB32(0, 1, 0, 1)  # Green
        elseif game.state[i, j] == 2            # Food
            img[i, j] = ARGB32(1, 0, 0, 1)  # Red
        end
    end
    return plot(img)
end

# Function to place new food, not really random: the seed is fixed at the beginning of a new episode.

function sample_food!(game::SnakeGame)

    # Find empty positions
    empty_positions = [(i, j) for i in 2:(game.state_size-1), j in 2:(game.state_size-1) if game.state[i, j] == 0]
    
    if isempty(empty_positions)
        return  # No space left
    end

    # Pick a random empty spot
    food_pos = rand(game.rng, empty_positions)
    
    println(food_pos)
    
    # Update game state
    game.state[food_pos...] = 2
    game.prev_food = food_pos
end 

#function to move the snake
function move!(game::SnakeGame, direction::String)
    
    if direction == "left"
    
    elseif direction == "right"
    
    elseif direction == "up"
    
    elseif direction == "down"
    
    end
	

end


# ----------------------- Test -------------------------------
game = SnakeGame()
sample_food!(game)  # Place new food
plot_state(game)    # Visualize game
                 

