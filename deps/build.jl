
try 
# construct absolute path
depsdir  = splitdir(Base.source_path())[1]
builddir = joinpath(depsdir,"builds")
srcdir   = joinpath(depsdir,"src")

println("=== Building Multigrid ===")
println("depsdir  = $depsdir")
println("builddir = $builddir")
println("srcdir   = $srcdir")

if !isdir(builddir)
	println("creating build directory")
	mkdir(builddir)
	if !isdir(builddir)
		error("Could not create build directory")
	end
end

@static if Sys.isunix()
	src1 = joinpath(srcdir,"Vanka.c")
	src2 = joinpath(srcdir,"parRelax.c");
	outfile1 = joinpath(builddir,"Vanka.so")
	outfile2 = joinpath(builddir,"parRelax.so")
	@build_steps begin
		println("gcc version")
		run(`gcc --version`)
		run(`gcc -O3 -fPIC -cpp -fopenmp -shared  $src1 -o $outfile1`)
		run(`gcc -O3 -fPIC -cpp -fopenmp -shared  $src2 -o $outfile2`)
	end
end

@static if Sys.iswindows() 
	src1 = joinpath(srcdir,"Vanka.c")
	src2 = joinpath(srcdir,"parRelax.c");
	outfile1 = joinpath(builddir,"Vanka.dll")
	outfile2 = joinpath(builddir,"parRelax.dll")
	#@build_steps begin
		println("gcc version")
		run(`gcc --version`)
		run(`gcc -O3 -cpp -fopenmp -shared -DBUILD_DLL  $src1 -o $outfile1`)
		run(`gcc -O3 -cpp -fopenmp -shared -DBUILD_DLL  $src2 -o $outfile2`)
	#end
end
catch
	println("Multigrid::build: Unable to build Multigrid")
end



