@testset "Multigrid.Multigrid" begin
    @testset "testSAforDivSigGrad.jl" begin
       include("testSAforDivSigGrad.jl")
    end
    @testset "testGMGRAPforPoisson.jl" begin
       include("testGMGRAPforPoisson.jl")
    end
    @testset "testGMG.jl" begin
      include("testGMG.jl")
    end
    @testset "testLinSolveMGWrapper.jl" begin
        include("testLinSolveMGWrapper.jl")
    end
	@testset "testGMGRAPforElasticity.jl" begin
       include("testGMGRAPforElasticity.jl")
    end
	# @testset "testHybridKaczmarz.jl" begin
       # include("testHybridKaczmarz.jl")
    # end
	
end