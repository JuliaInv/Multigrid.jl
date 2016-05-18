export SpMatMul,addVectors


function SpMatMul(alpha::Union{Float64,Complex128,Complex64},AT::SparseCSCTypes,x::ArrayTypes,beta::Union{Float64,Complex128,Complex64},target::ArrayTypes,numCores::Int64)
# function calculates: target = beta*target + alpha*A*x
if hasParSpMatVec
	Ac_mul_B!( alpha, A, x, beta, target, numCores );
else
	Base.Ac_mul_B!(alpha,A,x,beta,target);
end

return target;
end


function SpMatMul(AT::SparseCSCTypes,x::ArrayTypes,target::ArrayTypes,numCores::Int64)
alpha = one(eltype(x));
beta = zero(eltype(x));
if hasParSpMatVec
	Ac_mul_B!( alpha, A, x, beta, target, numCores );
else
	Base.Ac_mul_B!(alpha,A,x,beta,target);
end
return target;
end


function addVectors(alpha::Union{Float64,Complex128,Complex64},x::ArrayTypes,target::ArrayTypes)
# target <- target + alpha*x;
BLAS.axpy!(length(x), alpha, x,1, target, 1);
return;
end