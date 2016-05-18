export SpMatMul,addVectors


function SpMatMul(alpha::Union{Float64,Complex128,Complex64},A::SparseCSCTypes,x::ArrayTypes,beta::Union{Float64,Complex128,Complex64},target::ArrayTypes,numCores::Int64)
# function calculates: target = beta*target + alpha*A*x
# Base.Ac_mul_B!(alpha,A,x,beta,target);
ParSpMatVec.Ac_mul_B!( alpha, A, x, beta, target, numCores );
return target;
end


function SpMatMul(A::SparseCSCTypes,x::ArrayTypes,target::ArrayTypes,numCores::Int64)

alpha = one(eltype(x));
beta = zero(eltype(x));
# Base.Ac_mul_B!(alpha,A,x,beta,target);
ParSpMatVec.Ac_mul_B!( alpha, A, x, beta, target, numCores );
return target;
end


function addVectors(alpha::Union{Float64,Complex128,Complex64},x::ArrayTypes,target::ArrayTypes)
# target <- target + alpha*x;
BLAS.axpy!(length(x), alpha, x,1, target, 1);
return;
end