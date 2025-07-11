#debug the new environment with double states.
#setting DEBUG LEVEL
ENV["JULIA_DEBUG"] = Main

include("imports.jl")

"""
#die immediatelly turning left
gif_name = "double_die_now"
actions_list = [CartesianIndex(0,-1)]

game, exp_vec, episode_reward, anim = play_episode_with_animation(actions_list; gif_name = gif_name)
"""

#eating queue
gif_name = "double_eating_queue"
actions_list=[
    CartesianIndex(-1, 0), CartesianIndex(-1, 0), CartesianIndex(-1, 0),
    CartesianIndex(-1, 0), CartesianIndex(0, 1), CartesianIndex(0, 1),
    CartesianIndex(0, 1), CartesianIndex(1,0), CartesianIndex(1,0), 
    CartesianIndex(1,0), CartesianIndex(0,1),
    CartesianIndex(0,1), CartesianIndex(-1,0), CartesianIndex(-1,0),
    CartesianIndex(0,-1), CartesianIndex(1,0), CartesianIndex(0,1)
    ]

game, exp_vec, episode_reward, anim = play_episode_with_animation(actions_list; gif_name = gif_name)
