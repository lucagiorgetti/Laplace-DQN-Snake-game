include("env.jl")
using Printf

#the am overwriting the default first action, which is move up.
 
actions = ['.', CartesianIndex(-1,0), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(0,-1)]
game = SnakeGame()

plt = plot_or_update!(game)  # Plot the initial frame before the loop
sample_food!(game)

anim = @animate for (i, act) in enumerate(actions)
    if i > 1
        game.direction = act
        move_wrapper!(game)
    end
    
    plot_or_update!(game, plt)  # Update the existing plot

    if game.lost
        @printf "Collision!! Final score %d, iteration %d. \n" game.score i
        break
    end
end 

gif(anim, "./gifs/collision.gif", fps=1)

#eating food
actions = ['.', CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(-1,0), CartesianIndex(-1,0)]
 
plt = nothing
game = SnakeGame()
plt = plot_or_update!(game)  # Plot the initial frame before the loop
sample_food!(game)

anim = @animate for (i, act) in enumerate(actions)
    if i > 1
        game.direction = act
        move_wrapper!(game)
    end
    
    plot_or_update!(game, plt)  # Update the existing plot

    if game.lost
        @printf "Collision!! Final score %d, iteration %d. \n" game.score i
        break
    end
end 

gif(anim, "./gifs/eating_food.gif", fps=1)

#snake eating itself
actions = ['.', CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,-1), CartesianIndex(-1,0)]
 
plt = nothing
game = SnakeGame()
plt = plot_or_update!(game)  # Plot the initial frame before the loop
sample_food!(game)

anim = @animate for (i, act) in enumerate(actions)
    if i > 1
        game.direction = act
        move_wrapper!(game)
    end
    
    plot_or_update!(game, plt)  # Update the existing plot
    
    if game.lost
        @printf "Collision!! Final score %d, %d. \n" game.score i
        break
    end
end 

gif(anim, "./gifs/eating_itself.gif", fps=1)

#eating two foods
actions = ['.', CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(0,-1)]
 
plt = nothing
game = SnakeGame()
plt = plot_or_update!(game)  # Plot the initial frame before the loop
sample_food!(game)

anim = @animate for (i, act) in enumerate(actions)
    if i > 1
        game.direction = act
        move_wrapper!(game)
    end
    
    plot_or_update!(game, plt)  # Update the existing plot

    if game.lost
        @printf "Collision!! Final score %d, iteration %d. \n" game.score i
        break
    end
end 

gif(anim, "./gifs/eating_two_foods.gif", fps=1)

#eating three foods
actions = ['.', CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0),CartesianIndex(1,0), CartesianIndex(0,1)]
 
plt = nothing
game = SnakeGame()
plt = plot_or_update!(game)  # Plot the initial frame before the loop
sample_food!(game)

anim = @animate for (i, act) in enumerate(actions)
    if i > 1
        game.direction = act
        move_wrapper!(game)
    end
    
    plot_or_update!(game, plt)  # Update the existing plot

    if game.lost
        @printf "Collision!! Final score %d, iteration %d. \n" game.score i
        break
    end
end 

gif(anim, "./gifs/eating_three_foods.gif", fps=1)

#eating four foods
actions = ['.', CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0),CartesianIndex(1,0), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(0,1)]
 
plt = nothing
game = SnakeGame()
plt = plot_or_update!(game)  # Plot the initial frame before the loop
sample_food!(game)

anim = @animate for (i, act) in enumerate(actions)
    if i > 1
        game.direction = act
        move_wrapper!(game)
    end
    
    plot_or_update!(game, plt)  # Update the existing plot

    if game.lost
        @printf "Collision!! Final score %d, iteration %d. \n" game.score i
        break
    end
end 

gif(anim, "./gifs/eating_four_foods.gif", fps=1)

#almost eating itself
actions = ['.', CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0),CartesianIndex(1,0), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(0,-1)]

plt = nothing
game = SnakeGame()
plt = plot_or_update!(game)  # Plot the initial frame before the loop
sample_food!(game)

anim = @animate for (i, act) in enumerate(actions)
    if i > 1
        game.direction = act
        move_wrapper!(game)
    end
    
    plot_or_update!(game, plt)  # Update the existing plot

    if game.lost
        @printf "Collision!! Final score %d, iteration %d, length snake %d. \n" game.score i length(game.snake)
        for cc in game.snake
                println(cc)
        end
        break
    end
end 

gif(anim, "./gifs/almost_eating_itself.gif", fps=1)

#eating itself 2
actions = ['.', CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0),CartesianIndex(1,0), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(1,0), CartesianIndex(0,1)]

plt = nothing
game = SnakeGame()
plt = plot_or_update!(game)  # Plot the initial frame before the loop
sample_food!(game)

anim = @animate for (i, act) in enumerate(actions)
    if i > 1
        game.direction = act
        move_wrapper!(game)
    end
    
    plot_or_update!(game, plt)  # Update the existing plot

    if game.lost
        @printf "Collision!! Final score %d, iteration %d, length snake %d. \n" game.score i length(game.snake)
    break
    end
end 

gif(anim, "./gifs/eating_itself2.gif", fps=1)
