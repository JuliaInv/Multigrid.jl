export SpMatMul,addVectors


function SpMatMul(alpha::Union{Float64,ComplexF64,ComplexF32,Float32},AT::SparseCSCTypes,x::ArrayTypes,beta::Union{Float64,ComplexF64,ComplexF32},target::ArrayTypes,numCores::Int64)
# function calculates: target = beta*target + alpha*A*x
if hasParSpMatVec
	ParSpMatVec.Ac_mul_B!( alpha, AT, x, beta, target, numCores );
else
	mul!(target,adjoint(AT),x,alpha,beta);
end

return target;
end


function SpMatMul(AT::SparseCSCTypes,x::ArrayTypes,target::ArrayTypes,numCores::Int64)
alpha = one(eltype(x));
beta = zero(eltype(x));

if hasParSpMatVec
	ParSpMatVec.Ac_mul_B!( alpha, AT, x, beta, target, numCores );
else
	mul!(target,adjoint(AT),x,alpha,beta);
end
return target;
end


function addVectors(alpha::Union{Float64,ComplexF64,ComplexF32},x::ArrayTypes,target::ArrayTypes)
# target <- target + alpha*x;
if eltype(x) == eltype(target)
	BLAS.axpy!(length(x), alpha, x,1, target, 1);
else
	target .+= alpha*x;
end
return;
end