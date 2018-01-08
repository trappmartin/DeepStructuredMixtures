function convertToSPN(rootRegion::SumRegion, gpRegions, X, y, meanFunction, kernelFunction, noise; sampleNoise = false)
    
    nodes = Dict{Int, Vector{SPNNode}}()
    
    
    for r in gpRegions
    
        s = (X .> r.min) .& (X .<= r.max)
        xx = X[s]
        yy = y[s]

        # build start GP
        gp = GP(reshape(xx, 1, length(xx)), yy, meanFunction, kernelFunction, noise)

        # start mcmc
        chain = if sampleNoise
            mcmc(gp, burnin = 100, nIter = 1100)
        else
            mcmcKernel(gp, burnin = 100, nIter = 1100)
        end

        # get samples (random)
        samples = rand(1:1000, numGPs)
        while length(unique(samples)) != numGPs
            samples = rand(1:1000, numGPs)
        end

        # construct GPs
        gps = []
        for sample in samples
            
            ns = if sampleNoise
                chain[3,sample]
            else
                noise
            end
            
            node = GPLeaf{Any}(nextID(), 
                GP(reshape(xx, 1, length(xx)), yy, MeanZero(), SE(chain[1,sample],chain[2,sample]), ns))
            node.parents = SPNNode[]
            push!(gps, node)
        end

        nodes[RegionIDs[r]] = gps
    end
    
    return buildNodes(rootRegion, nodes, rootRegion)[1]
end

function buildNodes(r::SumRegion, nodes::Dict, rootRegion::SumRegion)
    if !haskey(nodes, RegionIDs[r])
        
        childrn = reduce(vcat, map(p -> buildNodes(p, nodes, rootRegion), r.partitions))

        if r == rootRegion
            # construct only a single sum node
            node = GPSumNode(nextID(), Int[]);

            for child in childrn
                add!(node, child)
            end

            fill!(node.prior_weights, 1. / length(node))
            fill!(node.posterior_weights, 1. / length(node))
            
            @assert length(node) == length(node.posterior_weights)
            
            nodes[RegionIDs[r]] = [node]
        else
            n = SPNNode[]
            for s in 1:numSums

                # construct only a single sum node
                node = GPSumNode(nextID(), Int[]);

                for child in childrn
                    add!(node, child)
                end

                fill!(node.prior_weights, 1. / length(node)) # use Dirichlet instead ?
                fill!(node.posterior_weights, 1. / length(node))
                
                @assert length(node) == length(node.posterior_weights)
                push!(n, node)
            end
            nodes[RegionIDs[r]] = n
        end
    end
    
    return nodes[RegionIDs[r]]
end

function buildNodes(p::SplitPartition, nodes::Dict, rootRegion::SumRegion)

    childrn = map(r -> buildNodes(r, nodes, rootRegion), p.regions)
    
    n = SPNNode[]
    for ch1 in childrn[1]
        for ch2 in childrn[2]
            # construct node
            node = FiniteSplitNode(nextID(), Float64[p.split])
        
            add!(node, ch1)
            add!(node, ch2)
            push!(n, node)
        end
    end
    
    node = GPSumNode(nextID(), Int[]);

    for child in n
        add!(node, child)
    end

    fill!(node.prior_weights, 1. / length(node)) # use Dirichlet instead ?
    fill!(node.posterior_weights, 1. / length(node))
    push!(n, node)
    
    nodes[PartitionIDS[p]] = [node]
    return nodes[PartitionIDS[p]]
end

function buildNodes(r::GPRegion, nodes::Dict, rootRegion::SumRegion)
    return nodes[RegionIDs[r]]
end
    
    