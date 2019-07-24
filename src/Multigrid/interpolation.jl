export getInterpolation

function getInterpolation(AT::SparseMatrixCSC, S::SparseMatrixCSC, coloring::Array, n::Int64)
	S.nzval .= 1.0
    S .= AT .* S
	## Pp - P.colptr
	## P.j - P.rowval
	## Here we build PT. 
	Pp = getInterpolation1(AT, S, coloring, n) ## Essentially counts non-zeros per row.
	Pp .= Pp .+ 1
	#(Pp, Pj, Px) = getInterpolation2(AT, S, Pp, coloring, n)
	(Pp, Pj, Px) = getDirectInterpolation2(AT, S, Pp, coloring, n)
	Pj .= Pj .+ 1
	R = SparseMatrixCSC(maximum(Pj), n, Pp, Pj, Px)
	P = sparse(R')
	return P, R
end

function getInterpolation1(AT::SparseMatrixCSC, S::SparseMatrixCSC, coloring::Array, n::Int64)
	Pp = zeros(Int, size(AT.colptr))
	nz = 0
	for i=1:n
		if (coloring[i] == 1)
			nz += 1
		else
			for j in nzrange(S, i)
				if S.rowval[j] != i && coloring[S.rowval[j]] == 1
					nz += 1
				end
			end
		end
		Pp[i+1] = nz
	end
	return Pp
end


# Direct interpolation.
# A little faster
# Discussed here : https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/yang1.pdf
# and on "Multigrid" book.
# First one I implemented for testing. Implementation from PyAMG (originally in C++).

function getDirectInterpolation2(AT::SparseMatrixCSC, S::SparseMatrixCSC, Pp::Array, coloring::Array, n::Int64)
	Px = zeros(Float64, Pp[end] - 1)
	Pj = zeros(Int, Pp[end] - 1)

	for i = 1:n
		# Make coarse node the i row of identity
		if coloring[i] == 1
			Pj[Pp[i]] = i
			Px[Pp[i]] = 1
			continue
		end

		# get positive and negative sums of nodes that are strongly connected to i
		(sum_strong_pos, sum_strong_neg) = getStrongSum(S, coloring, i)
		# get sum a_ij for j in 1:n , split to positive and negative
		(sum_all_pos, sum_all_neg, a_ii) = getAllSum(AT, coloring, i)

		i_alpha = sum_all_neg / sum_strong_neg
		i_beta = sum_all_pos / sum_strong_pos

		if sum_strong_pos == 0
			a_ii += sum_all_pos
			i_beta = 0.0
		end

		neg = -1 * i_alpha / a_ii
		pos = -1 * i_beta / a_ii

		nz = Pp[i]

		for j in nzrange(S, i)
			if coloring[S.rowval[j]] == 1 && S.rowval[j] != i
				Pj[nz] = S.rowval[j]
				if S.nzval[j] > 0
					Px[nz] = pos * S.nzval[j]
				else
					Px[nz] = neg * S.nzval[j]
				end
				nz += 1
			end
		end
	end

	# for node i in F : Pi = sum(w_ij*e_j). Here we have the transpose
	sum_map = zeros(Int, n)
	p_sum = 0
	for i = 1:n
		sum_map[i] = p_sum
		p_sum += coloring[i]
	end
	Pj .= sum_map[Pj]

   return Pp, Pj, Px
end


# get sum(a_ii + sum(a_in)) for n in weakly connected set
function getDenominator(AT::SparseMatrixCSC, S::SparseMatrixCSC, i::Int64)
	denominator = 0
	for j in nzrange(AT, i)
		denominator += AT.nzval[j]
	end

	for j in nzrange(S,i)
		if S.rowval[j] != i
			denominator -= S.nzval[j]
		end
	end
	return denominator
end

# Sum for strong coarse neighbours, returns tow sums : one for positive elements, and the other for negative elements
function getStrongSum(S::SparseMatrixCSC, coloring::Array, i::Int64)
	sum_positive = 0.0
	sum_negative = 0.0
	for j in nzrange(S, i)
		if coloring[S.rowval[j]] == 1 && S.rowval[j] != i
			if S.nzval[j] > 0
				sum_positive += S.nzval[j]
			else
				sum_negative += S.nzval[j]
			end
		end
	end
	sum_positive, sum_negative
end


function getAllSum(AT::SparseMatrixCSC, coloring::Array, i::Int64)
	sum_all_pos = 0.0
	sum_all_neg = 0.0
	diag = 0.0
	for j in nzrange(AT, i)
		if AT.rowval[j] == i
			diag += AT.nzval[j]
		else
			if AT.nzval[j] < 0
				sum_all_neg += AT.nzval[j]
			else
				sum_all_pos += AT.nzval[j]
			end
		end
	end
	sum_all_pos, sum_all_neg, diag
end



function getInnerDenominator(S::SparseMatrixCSC, coloring::Array, i::Int64, m::Int64)
	inner_denominator = 0
	# inner_denominator is sum of a_mk such that:
	# k is in C_i (coarse strong connection) and m is in F_i (fine strong connection)
	for k in nzrange(S,S.rowval[m])
		for kk in nzrange(S,i)
			if S.rowval[k] == S.rowval[kk] && coloring[S.rowval[k]] == 1
				inner_denominator+=S.nzval[k]
			end
		end
	end
	return inner_denominator
end



## This is the textbook version, discussed in "A Multigrid Tutorial".
# We calculate the i component of the operator :
# For every coarse node, the row is unity (e_i)
# For every free node, we calculate the weight and put sum(w_ij * e_j)

function getInterpolation2(AT::SparseMatrixCSC, S::SparseMatrixCSC, Pp::Array, coloring::Array, n::Int64)

	Px = zeros(Float64, Pp[end] - 1)
	Pj = zeros(Int, Pp[end] - 1)

	for i=1:n
		# Make coarse node the i row of identity
		## coloring = 1: C node, 0: F node
		if coloring[i] == 1
			Pj[Pp[i]] = i
			Px[Pp[i]] = 1
			continue
		end
		nz = Pp[i]
		# denominator holds all a_ij for weakly connected i-j.
		# with sparse arrays it's easier to take all, and subtract strong connections.
		i_denominator = getDenominator(AT, S, i)

		assert(i_denominator != 0)
		#j_numerator : AT[j][i] + ( AT[m][i] * AT[j][m] /  ( AT[k][m] for k in nz(S,i) if coloring(k) == 1) for m in nz(S,i) if coloring (k == 0))

		for j in nzrange(S,i)
			if S.rowval[j] == i || coloring[S.rowval[j]] == 0
				continue
			end

			Pj[nz] = S.rowval[j]

			j_numerator = S.nzval[j]
			for m in nzrange(S,i)
				# Ignore strong coarse connections
				if S.rowval[m] == i || coloring[S.rowval[m]] == 1
					continue
				end
				# check that AT[m][i], AT[j][m] != 0, otherwise there's no contribution to the numerator.
				inner_denominator = getInnerDenominator(S, coloring, i, m)
				assert(inner_denominator != 0)
				for l in nzrange(AT, S.rowval[m])
					if AT.rowval[l] == S.rowval[j]
						j_numerator += (S.nzval[m] * AT.nzval[l]) / inner_denominator
					end
				end
			end
			Px[nz] = - (j_numerator / i_denominator)
			nz += 1
		end
	end

	# for node i in F : Pi = sum(w_ij*e_j). Here we have the transpose.
	sum_map = zeros(Int, n)
	p_sum = 0
	for i = 1:n
		sum_map[i] = p_sum
		p_sum += coloring[i]
	end
	Pj .= sum_map[Pj]
	return Pp, Pj, Px
end