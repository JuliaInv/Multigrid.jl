@testset "Multigrid.Multigrid" begin
    @testset "testGMG.jl" begin
      include("testGMG.jl")
    end
    @testset "testGMGRAPforPoisson.jl" begin
       include("testGMGRAPforPoisson.jl")
    end
	@testset "testSAforDivSigGrad.jl" begin
		include("testSAforDivSigGrad.jl")
    end
    @testset "testLinSolveMGWrapper.jl" begin
        include("testLinSolveMGWrapper.jl")
    end
	@testset "testLinSolveAMGWrapper.jl" begin
        include("testLinSolveAMGWrapper.jl")
    end
	@testset "testGMGRAPforElasticity.jl" begin
       include("testGMGRAPforElasticity.jl")
    end
	@testset "testGMGforElasticity.jl" begin
       include("testGMGforElasticity.jl")
    end
	@testset "testGMGRAPforElasticityVanka.jl" begin
        include("testGMGRAPforElasticityVanka.jl")
    end
	@testset "testGMGforElasticityVanka.jl" begin
       include("testGMGforElasticityVanka.jl")
    end
	@testset "testHybridKaczmarz.jl" begin
       include("testHybridKaczmarz.jl")
    end
end