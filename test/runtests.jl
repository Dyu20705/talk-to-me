using Test
using var"Talk-to-Me"

@testset "Talk-to-Me" begin
    @test greet("Duy") == "Hello, Duy! Welcome to Talk-to-Me."
end
