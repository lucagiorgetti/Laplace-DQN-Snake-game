#the replay buffer stores touples of the form (state, action, reward, next_state, done_or_not::Bool), position tracks the index where the next experience must be stored.
using StatsBase

mutable struct ReplayBuffer
        capacity::Int
        position::Int
        buffer::Vector{Tuple{Matrix{Int64}, CartesianIndex{2}, Float64, Matrix{Int64}, Bool}}
        
        function ReplayBuffer(capacity = 10000)
                  buffer = Vector{Tuple{Matrix{Int64}, CartesianIndex{2}, Float64, Matrix{Int64}, Bool}}(undef, 0)
                  new(capacity, 1, buffer)
        end
end

#definition length of a buffer
Base.length(rpb::ReplayBuffer) = length(rpb.buffer)

function store!(rpb::ReplayBuffer, exp::Tuple{Matrix{Int64}, CartesianIndex{2}, Float64, Matrix{Int64}, Bool})
          if length(rpb) < rpb.capacity
              push!(rpb.buffer, exp)
          else
             rpb.buffer[rpb.position] = exp
             rpb.position += 1 
          end
          if rpb.position > rpb.capacity
              rpb.position = 1
          end
end

#I think sampling without replacement is faster because I do not want to repeat the initial state-actions for too many times.
function sample(rpb::ReplayBuffer, dim::Int)
          if length(rpb) < dim
              return sample(rpb.buffer, length(rpb), replace = false)
          else
              return sample(rpb.buffer, dim, replace = false)
          end 
end

function check_full(rpb::ReplayBuffer)
          return rpb.position == rpb.capacity
end
