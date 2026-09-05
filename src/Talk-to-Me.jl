module var"Talk-to-Me"

using Flux
using JSON3
using DataFrames
using LinearAlgebra
using Random

export greet

function greet(name::AbstractString="User")
    return "Hello, $(name)! Welcome to Talk-to-Me."
end

end
