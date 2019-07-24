export getColoringFirst
export getColoringSecond
export getColoringSecondS



# Make a maximal independent set of coarse nodes from the grid.
#
# The slowest part of the naive algorithm is taking max(lambda) over and over again
# A simple solution is updating the maximum as we increase values in lambda
# When no index of lambda has increased over the maximum, we need to check if the maximum is now lower

function getColoringFirst(S::SparseMatrixCSC, n::Int64)

	# lambda - count of strongly connected nodes to current
	# coloring -  will hold the C/F value for each node. C == 1, F == 0
	lambda = [0 for i=1:n]
	coloring = fill(0, n)
	rows = rowvals(S)

	# for each neighbour count, holds the nodes that has that number of neighbours.
	indeces = [Set{Int}() for i=1:n+1]

	# store by neigbour count, store max
	nmax = 0
	for i = 1:n
		lambda[i] = S.colptr[i+1] - S.colptr[i]
	end
	for i=1:n
		# Has only diagonal, make it free
		if(lambda[i] == 1)
			lambda[i] == 0
			coloring[i] == 0
		end
		push!(indeces[lambda[i] + 1],i)
		if (lambda[i] > nmax)
			nmax = lambda[i]
		end
	end

	while (nmax > 0)
		old_max = nmax

		# take a node that has a maximal neigbour count
		curr = pop!(indeces[nmax + 1])

		# set the node as coarse, and reset it's neigbour count
		coloring[curr] = 1
		lambda[curr] = 0

		# look at every neigbour it has
		for j in nzrange(S,curr)
			row = rows[j]
			if lambda[row] != 0
				# mark as free node
				pop!(indeces[lambda[row] + 1], row)
				lambda[row] = 0
				push!(indeces[lambda[row] + 1], row)
				coloring[row] = 0
			end
		end

		for j in nzrange(S,curr)
			row = rows[j]
			for k in nzrange(S, row)
				rowk = rows[k]
				# node was already set, ignore it.
				if (lambda[rowk] == 0)
					continue
				end

				# update node count
				pop!(indeces[lambda[rowk] + 1], rowk)
				lambda[rowk] += 1
				push!(indeces[lambda[rowk] + 1], rowk)


				# we have a new max
				if(lambda[rowk] > nmax)
					nmax = lambda[rowk]
				end
			end

		end

		# No new max, check if max should be lower
		if (old_max == nmax)
			for j=nmax:-1:0
				nmax = j
				if (~isempty(indeces[j + 1]))
					break
				end
			end
		end
	end
	coloring
end



# Change coloring according to heuristic:
# for every strongly connected F-F couple,
# both nodes has to have at least one common strongly connected C node.
function getColoringSecond(S::SparseMatrixCSC, coloring::Array, n::Int64)
	for i=1:n
		# This is a coarse node
		if coloring[i] == 1
			continue
		end
		# Split nodes strongly connected to i, into F and C sets
		fconn, cconn = strongSplitting(S, coloring, i)
		# Go through every F-F connection
		for j in fconn
			# if we don't have a common C node, make current node a C node
			if !hasCommonC(S, coloring, cconn, i, j)
				coloring[i] = 1
				break
			end
		end
	end
	return coloring
end


# Check if nodes i,j has a common strongly connected C node
function hasCommonC(S::SparseMatrixCSC, coloring::Array, cconn::Set{Int}, i::Int64, j::Int64)
	# Check for common C nodes
	for k in nzrange(S, j)
		if S.rowval[k] == i
			continue
		end
		if coloring[S.rowval[k]] == 1 && S.rowval[k] in cconn
			# println("Found F-F with common C : $i , $j, C is $(S.rowval[k])")
			return true
		end
	end
	return false
end

# Split the nodes strongly connected to i into coarse a set (cconn) and a free set (fconn)
function strongSplitting(S::SparseMatrixCSC, coloring::Array, i::Int64)
	fconn = Set{Int}()
	cconn = Set{Int}()

	for j in nzrange(S,i)
		# skip if this is a node already checked, or is the current node (that means i)
		if S.rowval[j] == i
			continue
		end
		if coloring[S.rowval[j]] == 0
			push!(fconn, S.rowval[j])
		else
			push!(cconn, S.rowval[j])
		end
	end
	fconn, cconn
end






## Alternative algorithm. Tries to minimize the coarse grid.


# idea
# get all F-F links for a free node, which do not have a common C neighbour and store in set.
# Do so for each free node - will be saved in an Array.
# now take the node with the maximal number of such neighbours,
# remove it from all it's neighbours' sets, remove it's set and turn it into a coarse node.
# repeat until all sets are empty
function getColoringSecondS(S::SparseMatrixCSC, coloring::Array, n::Int64)
	(fconn, cconn) = getConnections(S, coloring, n)

	maxi = 0
	max_node = 1

	# fconn holds all F-F connections that doesn't have a common C node
	# Make the free node with the maximal amount of such neighbours, a C node. Until no more exist
	while(true)
		(fconn, maxi, max_node) = clearConnections(S, fconn, cconn, n, 0, 1)
		if maxi == 0
			break
		end

		coloring[max_node] = 1
		for i in fconn[max_node]
			pop!(fconn[i],max_node)
			push!(cconn[i],max_node)
		end
		empty!(fconn[max_node])
	end

	return coloring
end


# Get all strong connections for every node, split into a coarse set and a free set.
# fconn[i] holds all free nodes strongly connected to i
# cconn[i] holds all coarse nodes strongly connected to i
function getConnections(S::SparseMatrixCSC, coloring::Array, n::Int64)
	fconn = [Set{Int}() for i=1:n]
	cconn = [Set{Int}() for i=1:n]
	# Populate fconn with all F-F connections, and cconn with all C-F connections
	for i=1:n

		# Skip over coarse nodes
		if coloring[i] == 1
			continue
		end

		for j in nzrange(S, i)
			if i == S.rowval[j]
				continue
			end

			# coarse neighbour
			if coloring[S.rowval[j]] == 1
				push!(cconn[i],S.rowval[j])
				continue

			# free neighbour
			else
				push!(fconn[i],S.rowval[j])
			end

		end
	end
	fconn, cconn
end


# Remove all free nodes strongly connected, such that they have a common coarse node.
# fconn will be left with all F-F connections without a common node in cconn
function clearConnections(S::SparseMatrixCSC, fconn::Array{Set{Int64},1}, cconn::Array{Set{Int64},1}, n::Int64, maxi::Int64, max_node::Int64)
	for i=1:n
		for j in fconn[i]
			if i == j
				continue
			end
			for k in cconn[j]
				# Has a common c node, remove F-F connection
				if k in cconn[i]
					pop!(fconn[i],j)
					pop!(fconn[j],i)
					break
				end
			end
		end
		if length(fconn[i]) > maxi
			max_node = i
			maxi = length(fconn[i])
		end
	end
	return fconn, maxi, max_node
end