
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
	src3 = joinpath(srcdir,"parLU.cpp");
	outfile1 = joinpath(builddir,"Vanka.so")
	outfile2 = joinpath(builddir,"parRelax.so")
	outfile3 = joinpath(builddir,"parLU.so")
	println("gcc version")
	run(`gcc --version`)
	run(`gcc -O3 -fPIC -cpp -fopenmp -shared  $src1 -o $outfile1`)
	run(`gcc -O3 -fPIC -cpp -fopenmp -shared  $src2 -o $outfile2`)
	run(`g++ -O3 -fPIC -cpp -fopenmp -shared  $src3 -o $outfile3`)
end

@static if Sys.iswindows() 
	src1 = joinpath(srcdir,"Vanka.c")
	src2 = joinpath(srcdir,"parRelax.c");
	src3 = joinpath(srcdir,"parLU.cpp");
	outfile1 = joinpath(builddir,"Vanka.dll")
	outfile2 = joinpath(builddir,"parRelax.dll")
	outfile3 = joinpath(builddir,"parLU.dll")
	## This is needed for Cygwin
	src1 = replace(src1, "\\" => "/")
	src2 = replace(src2, "\\" => "/")
	src3 = replace(src3, "\\" => "/")
	
	#println(string("gcc -O3 -cpp -fopenmp -shared -DBUILD_DLL"," ",src1," -o ",outfile1)) 
	println("gcc version")
	run(`gcc --version`)
	
	run(`gcc -O3 -cpp -fopenmp -shared -DBUILD_DLL  $src1 -o $outfile1`)
	run(`gcc -O3 -cpp -fopenmp -shared -DBUILD_DLL  $src2 -o $outfile2`)
	run(`g++ -O3 -cpp -fopenmp -shared -DBUILD_DLL  $src3 -o $outfile3`)
end
catch 
	@warn "Multigrid::build: Unable to build Multigrid"
end



