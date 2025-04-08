include("env.jl")
using Printf

function animate_snake(actions::Vector{Any}, gif_name::String)
         plt = nothing
         game = SnakeGame(10,1)
         plt = plot_or_update!(game)
         sample_food!(game)
         
         break_next = false
         
         anim = @animate for (i, act) in enumerate(actions)
             if i > 1 && break_next == false
                 game.direction = act
                 move_wrapper!(game)
             end
             
             if break_next == true
                break
             end 
             
             plot_or_update!(game, plt)
             
             if game.lost
                 @printf "Collision!! Final score %d, reward %.3f \n" game.score game.reward
                 break_next = true
             end
         end
         
         gif(anim, "./gifs/"*gif_name*".gif", fps=1)
         return nothing              
end 

#I am overwriting the default first action, which is move up.
name_seq = ["collision", "eating_food", "eating_itself", "almost_eating_itself", "eating_queue"]

actions_seq = [['.', CartesianIndex(-1,0), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(0,-1)],

['.', CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(-1,0), CartesianIndex(-1,0)],

['.', CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,-1), CartesianIndex(-1,0)],

['.', CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0),CartesianIndex(1,0), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(0,-1)],

['.', CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(0,-1), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0), CartesianIndex(1,0),CartesianIndex(1,0), CartesianIndex(0,1), CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(-1,0), CartesianIndex(0,-1), CartesianIndex(1,0), CartesianIndex(1,0),
CartesianIndex(0,1), CartesianIndex(1,0)]]

for (name, actions) in zip(name_seq, actions_seq)
     animate_snake(actions, name)
end
