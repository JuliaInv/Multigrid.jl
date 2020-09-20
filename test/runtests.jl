using Test
@testset "Multigrid" begin
    include("Multigrid/runtests.jl")
	include("ParallelJuliaSolver/runtests.jl")
	include("DomainDecomposition/runtests.jl")
end



