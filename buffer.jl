#the replay buffer stores touples of the form (state, action, reward, next_state, done_or_not::Bool)

mutable struct ReplayBuffer
        capacity::Int
        position::Int
        buffer::Vector{Tuple{Matrix{Int64}, CartesianIndex{2}, Float64, Matrix{Int64}, Bool}}
        
        function ReplayBuffer(capacity = 10000)
                  buffer = Vector{Tuple{Matrix{Int64}, CartesianIndex{2}, Float64, Matrix{Int64}, Bool}}(undef, 0)
                  new(capacity, 1, buffer)
        end
end
