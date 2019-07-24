using jInv.Mesh
using ForwardHelmholtz
using ForwardHelmholtz.ElasticHelmholtz
using DomainDecomposition
using Multigrid
using KrylovMethods
domain = [0.0, 1.0, 0.0, 1.0];
n = [128,128];
Mr = getRegularMesh(domain,n)
numCores = 8;
innerKatzIter = 200;
omega = 4.0*getMaximalFrequency(ones(10,1),Mr);
spValType = Complex64;
gamma = 0.25;
inner = 5;



# m = ones(tuple(Mr.n+1...));
# # AcT = convert(SparseMatrixCSC{spValType,Int64},GetHelmholtzOperator(Mr, m,omega , 0.0001*m,false,[10,10],omega*1.0,false)[1])';
# McT = convert(SparseMatrixCSC{spValType,Int64},GetHelmholtzOperator(Mr, m,omega , omega*gamma*m,false,[10,10],omega*1.0,false)[1])';
# q = rand(spValType,size(McT,2));
# x = zeros(spValType,size(McT,2));

# HKparam = getHybridKaczmarz(McT,Mr,[8,8], getNodalIndicesOfCell,1.1,numCores,innerKatzIter);
# prec = getHybridKaczmarzPrecond(HKparam,McT,q);

# y = zeros(spValType,size(McT,2));
# # y = FGMRES(McT,copy(q),y,inner,prec,1e-5*norm(q),true,true,numCores)[1]
# println("Norm is: ",norm(McT'*y - q))



# println("Elastic Experiment")
# gamma = 0.25;
# m = ones(tuple(Mr.n...));
# EHparam = ElasticHelmholtzParam(Mr,omega,2*m,m,m,gamma*m,false,false);
# McT = convert(SparseMatrixCSC{spValType,Int64},GetElasticHelmholtzOperator(EHparam))';
# q = rand(spValType,size(McT,2));
# x = zeros(spValType,size(McT,2));
# HKparam = getHybridKaczmarz(McT,Mr,[8,8], getFacesStaggeredIndicesOfCellNoPressure,0.85,numCores,innerKatzIter);
# prec = getHybridKaczmarzPrecond(HKparam,McT,q);
# y = zeros(spValType,size(McT,2));
# z = prec(q);
# println("Norm is: ",norm(McT'*z - q)/norm(q))

# y = FGMRES(McT,copy(q),y,inner,prec,1e-5*norm(q),true,true,numCores)[1]
# println("Norm is: ",norm(McT'*y - q))



innerKatzIter = 4;
println("Elastic Experiment Mixed")
gamma = 0.25;
m = ones(tuple(Mr.n...));
mixedFormulation = true;
Kaczmarz = false;
println("Kaczmarz = ",Kaczmarz);
println("Mixed = ",mixedFormulation);
EHparam = ElasticHelmholtzParam(Mr,omega,2*m,m,m,gamma*m,false,mixedFormulation);
McT = GetElasticHelmholtzOperator(EHparam);
McT = McT/norm(McT,1);
q = rand(spValType,size(McT,2));
# q[100] = 1.0;
x = zeros(spValType,size(McT,2));
HKparam = getHybridCellWiseParam(McT,Mr,[4,4],0.5,numCores,innerKatzIter,mixedFormulation,Kaczmarz);

McT = convert(SparseMatrixCSC{spValType,Int64},McT);
prec = getHybridCellWisePrecond(HKparam,McT,copy(q),mixedFormulation,Kaczmarz);

z = prec(q);
println("Norm is: ",norm(McT'*z - q)/norm(q))

y = zeros(spValType,size(McT,2));
tic()
y = FGMRES(McT,copy(q),y,6*inner,prec,1e-5*norm(q),true,true,numCores)[1];
toc()
println("Norm is: ",norm(McT'*y - q)/norm(q))
y = zeros(spValType,size(McT,2));
Afun = getAfun(McT,copy(y),numCores);
x = KrylovMethods.fgmres(Afun,q,inner;tol = 1e-5,maxIter = 6,M = prec,x = y)[1];
println("Norm is: ",norm(McT'*y - q)/norm(q))





println("The end");
