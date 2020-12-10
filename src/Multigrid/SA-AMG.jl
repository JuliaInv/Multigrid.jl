export SA_AMGsetup;

"""
Based on the paper 
Eran Treister and Irad Yavneh, Non-Galerkin Multigrid based on Sparsified Smoothed Aggregation. 
SIAM Journal on Scientific Computing, 37 (1), A30-A54, 2015.
"""
function SA_AMGsetup(AT::SparseMatrixCSC{VAL,IND},param::MGparam{VAL,IND},symm::Bool = true,nrhs::Int64 = 1,verbose::Bool=false) where {VAL,IND}
Ps = Array{SparseMatrixCSC{real(VAL),IND}}(undef,param.levels-1);
Rs = Array{SparseMatrixCSC{real(VAL),IND}}(undef,param.levels-1);
As = Array{SparseMatrixCSC{VAL,IND}}(undef,param.levels);

if symm == false
	error("not supported yet...");
end	

T_time = 0;	
N = size(AT,2);
As[1] = AT;
relaxPrecs = Array{Array{VAL}}(undef,param.levels-1);
Cop = nnz(AT);
for l = 1:(param.levels-1)
	if verbose
		T_time = time_ns();
	end
    AT = As[l];
    if param.relaxType=="Jac" || param.relaxType=="Jac-GMRES" || param.relaxType=="SPAI"
		relaxPrecs[l] = getRelaxPrec(AT,VAL,param.relaxType,param.relaxParam,[],false);
	else
		error("Unknown relaxation type !!!!");
	end
	P0 = getAggregation(AT,param.strongConnParam);
	Nc = size(P0,2);
	P0 = sparse(P0');
	if (size(P0,1)==size(P0,2))
		if verbose; println(string("Stopped Coarsening at level ",l)); end
		param.levels = l;
		As = As[1:l];
		Ps = Ps[1:l-1];
		Rs = Rs[1:l-1];
		relaxPrecs = relaxPrecs[1:l-1];
		break;
	else
		DAT = AT*spdiagm(0=>relaxPrecs[l]); # Julia's .* doesn't work for this multiplication. Only from the right.
		rhoDAT = min(norm(DAT,1),norm(DAT,Inf));
		PT = P0 - (1.33/rhoDAT)*P0*DAT;
		Rs[l] = sparse(PT'); # this is becasue we hold the transpose of the matrices and P = R' anyway here....
		Ps[l] = PT;

		Act = Ps[l]*AT*Rs[l];
		As[l+1] = Act;
		Cop = Cop + nnz(Act);
		#fprintf('setup level %3d, dim:%3dx%3d, nnz:%3d\n',nlevels+1,nc(1),nc(2),nnz(Ac))
		
		if verbose; println("MG setup: ",N," took:",(time_ns() - T_time)/1e+9); end;
		N = Nc;
	end
end
if verbose 
	T_time = time_ns();
	println("MG Setup: Operator complexity = ",Cop/nnz(As[1]));
end
As[end] = As[end] + 1e-8*norm(As[end],1)*sparse(1.0I,size(As[end],2),size(As[end],2));

defineCoarsestAinv(param,As[end]);

if verbose 
	println("MG setup coarsest: ",N,", done LU in ",(time_ns() - T_time)/1e+9);
end
param.As = As;
param.Ps = Ps;
param.Rs = Rs;
param.relaxPrecs = relaxPrecs;
param = adjustMemoryForNumRHS(param,nrhs,verbose);
return;
end

function getAggregation(AT::SparseMatrixCSC,strengthConnParam::Float64)
	if size(AT,2) <= 100
		return sparse(1.0I,size(AT,2),size(AT,2));
	end
	S = getStrengthMatrix(AT,strengthConnParam);
	aggr = neighborhoodAggregationNew(S);
	# println(aggr)
	return aggrArray2P(aggr);
end

function getStrengthMatrix(AT::SparseMatrixCSC,strengthConnParam::Float64)
S = -AT;
mm = 1e-16*maximum(S.nzval);
n = size(S,2);
for j = 1:n
	maxVal_j = mm;
	for gidx = S.colptr[j] : S.colptr[j+1]-1
		if S.nzval[gidx] > maxVal_j
			maxVal_j = S.nzval[gidx]; 
		end
	end
	scal_k = 1.0 ./maxVal_j;
	for gidx = S.colptr[j] : S.colptr[j+1]-1
		S.nzval[gidx]*=scal_k; 
	end
	for gidx = S.colptr[j] : S.colptr[j+1]-1
		if S.rowval[gidx] == j
			# We have a diagonal element.
			S.nzval[gidx] = 1.0;
		end
	end
	for gidx = S.colptr[j] : S.colptr[j+1]-1
		if S.nzval[gidx] <  strengthConnParam
			S.nzval[gidx] = 0.0;
		end
	end
end
S = S + S';
end


function neighborhoodAggregationNew(S::SparseMatrixCSC)
## S is symmetric. We look only at non-zero values here...  
tau = 3.0;
n = size(S,2);
aggr = zeros(Int64,n);

avg_sparsity = 0.0;
aux = zeros(n);
aux_count = zeros(Int64,n);
for k=1:n
	avg_sparsity += S.colptr[k+1] - S.colptr[k]; 
end
avg_sparsity /= n;

for k = 1:n
    if S.colptr[k+1] - S.colptr[k] > tau*avg_sparsity
		aux_count[k] = -1; 
	end
end
#########################################################################
for k = 1:n
    Neighbors_Aggregated_flag = false;
	if aux_count[k] == -1
		continue; # ignore...
	end
    for global_idx = S.colptr[k] : S.colptr[k+1]-1
        if aggr[S.rowval[global_idx]] != 0
            Neighbors_Aggregated_flag = true;
            break;
		end
    end
	if !Neighbors_Aggregated_flag
        for global_idx = S.colptr[k] : S.colptr[k+1]-1
			if aux_count[S.rowval[global_idx]] != -1
				aggr[S.rowval[global_idx]] = k; #conversion to Matlab's indices;
				aux_count[k]+=1;
			end
        end
    end
end
#########################################################################
for k = 1:n
    Neighbors_Aggregated_flag = false;
	if aux_count[k]!=-1
		continue;
	end
	aux_count[k] = 0;
    for global_idx = S.colptr[k] : S.colptr[k+1]-1
        if aggr[S.rowval[global_idx]] != 0
            Neighbors_Aggregated_flag = true;
            break;
		end
    end
	if !Neighbors_Aggregated_flag
        for global_idx = S.colptr[k] : S.colptr[k+1]-1
			aggr[S.rowval[global_idx]] = k; #conversion to Matlab's indices;
			aux_count[k]+=1;
        end
    end
end
#########################################################################
for k = 1:n
	chosen_score = 0.0;
	chosen = 0;
	if aggr[k] == 0
		for global_idx = S.colptr[k] : S.colptr[k+1]-1
			if aggr[S.rowval[global_idx]] > 0 # aggr can be negative here....					
                agg_of_neighbor = aggr[S.rowval[global_idx]];
				aux[agg_of_neighbor] += S.nzval[global_idx];
			end
			for global_idx = S.colptr[k] : S.colptr[k+1]-1
				if aggr[S.rowval[global_idx]]>0 # aggr can be negative here....
                    agg_of_neighbor = aggr[S.rowval[global_idx]];
                    if chosen_score < aux[agg_of_neighbor]/aux_count[agg_of_neighbor] 
						chosen_score = aux[agg_of_neighbor]/aux_count[agg_of_neighbor];
						chosen = agg_of_neighbor;
                        aux[agg_of_neighbor] = 0;
					end
                end
			end
            aggr[k] = -chosen; # we don't want the new aggregate to spread - use only original aggregate
		end
	end
end
#########################################################################
for k = 1:n
	if aggr[k] < 0.0
		aggr[k] = -aggr[k];
	end
end

return aggr;
end

function aggrArray2P(aggr)
	fine2coarse = zeros(Int64,size(aggr));
	n = length(aggr);
	I = findall(aggr .== 1:n);
	fine2coarse[I] = 1:length(I);
	aggr = fine2coarse[aggr];
	if sum(aggr.==0)>0
		error("nodes without aggregates");
	end
	P = sparse(collect(1:n),aggr,ones(n));
	return P;
end