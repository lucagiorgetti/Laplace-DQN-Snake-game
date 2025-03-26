# some tests on env.jl, remember that the first action is move up
include("env.jl")
using Printf

actions = [CartesianIndex(-1,0), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(0,-1)]         #list of Cartesian Indices
game = SnakeGame()

anim = @animate for act in actions
             plot_state(game)
             game.direction = act
             move_wrapper!(game)
             if game.lost
                 @printf "Collision!! Final score %d \n." game.score
                 sleep(5)
                 break
             end
end

gif(anim, "snake_test.gif", fps=2)
#needs proper initialization
