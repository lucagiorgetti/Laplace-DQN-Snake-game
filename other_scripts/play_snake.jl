import Base: include
Base.include(@__MODULE__, "imports.jl")

gm = SnakeGame()

WIDTH = 600
HEIGHT = 600
BACKGROUND = colorant"antiquewhite"

final_score = 0

#unit, dimension of a pixel
u = 60

#colors
wall_color = colorant"black"
snake_color = colorant"green"
apple_color = colorant"red"

#define the walls actors
walls = [
    Rect(0, 0, WIDTH, u),              # top wall
    Rect(0, HEIGHT - u, WIDTH, u),     # bottom wall
    Rect(0, 0, u, HEIGHT),             # left wall
    Rect(WIDTH - u, 0, u, HEIGHT)      # right wall
]

function draw(g::Game)
    # background and walls
    for wall in walls
        draw(wall, wall_color, fill=true)
    end
    
    apple_rect, snake_rect = board_to_rectangles(gm.board)
    
    # snake
    for part in snake_rect
        draw(part, snake_color, fill=true)
    end
    
    #apple
    for part in apple_rect
        draw(part, apple_color, fill=true)
    end
end

########################################################################################################
#transformation game.board coordinates -> GameZero.jl coordinates
#1. apple (x,y) -> ((x-1)u, (y-1)u, u, u)
#2. snake. I will draw the snake as composed with single pixels, so the transformation is the same as the one of the apple applied to single segments.
#########################################################################################################

function coord_transform(coords::Tuple{Int,Int})::NTuple{4, Int}
          x,y = coords
          #return((x-1)*u, (y-1)*u, u, u)
          return((y-1)*u,(x-1)*u, u, u)    #notice x and y are inverted
end

function board_to_actors(board::Matrix{Int})
	  
	  apple_pos = []
	  snake_pos = []
	  
	  for x in 1:size(board, 1)
              for y in 1:size(board, 2)
                  if board[x,y] == 2
                     push!(apple_pos, (x,y))
                  elseif board[x,y] == 1
                     push!(snake_pos, (x,y))
                  end
              end
          end    
          
          apple_actor = coord_transform.(apple_pos)
          snake_actor = coord_transform.(snake_pos)
          
          return apple_actor, snake_actor
end

function actors_to_rectangles(apple::Vector{Tuple{Int,Int,Int,Int}}, snake::Vector{Tuple{Int,Int,Int,Int}})
         
         apple_rect = [Rect(coords...) for coords in apple]
         snake_rect = [Rect(coords...) for coords in snake]
         
         return apple_rect, snake_rect
end

function board_to_rectangles(board::Matrix{Int})
          
          apple_actor, snake_actor = board_to_actors(board)
          apple_rect, snake_rect = actors_to_rectangles(apple_actor, snake_actor)
          
          return apple_rect, snake_rect
end

function on_key_down(g::Game, key)
   
          if g.keyboard.UP
             action = CartesianIndex(-1,0)
             
          elseif g.keyboard.DOWN
                 action = CartesianIndex(1,0)
             
          elseif g.keyboard.LEFT
                 action = CartesianIndex(0,-1)
             
          elseif g.keyboard.RIGHT
                 action = CartesianIndex(0,1)           
          end
          
          step!(gm, action)
          
          if gm.lost
              global final_score = gm.score
              global snake_game = deepcopy(gm)
              
              @printf "final_score: %d \n" gm.score
              error("Lost")
          end
 
end

