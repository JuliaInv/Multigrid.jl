@testset "Multigrid.DomainDecomposition" begin
	@testset "testDDPoisson.jl" begin
      include("testDDPoisson.jl")
    end
	@testset "testDDParallel_Poisson.jl" begin
      include("testDDParallel_Poisson.jl")
    end
end