function convertToSPN(rootRegion::SumRegion, gpRegions, X, y, meanFunction, kernelFunctions::Vector, noise; overlap = 2)
    
    nodes = Dict{Int, Vector{SPNNode}}()
    for r in gpRegions
    
        s = (X .> (r.min - overlap)) .& (X .<= (r.max + overlap))
        xx = X[s]
        yy = y[s]

        # construct GPs
        gp_nodes = []
        for kernel_function in kernelFunctions
               
            node = GPLeaf{Any}(nextID(), 
                GP(reshape(xx, 1, length(xx)), yy, deepcopy(meanFunction), deepcopy(kernel_function), deepcopy(noise))
                )
            node.gp = GaussianProcesses.fit!(node.gp, reshape(xx, 1, length(xx)), yy)
            node.parents = SPNNode[]
            push!(gp_nodes, node)
        end

        nodes[RegionIDs[r]] = gp_nodes
    end
    
    return buildNodes(rootRegion, nodes, rootRegion)[1]
end

function convertToSPN_ND(rootRegion::NDSumRegion, gpRegions, RegionIDs, PartitionIDS, X, y, meanFunction, kernelFunctions::Vector, kernelPriors::Vector, noise; overlap = 2, do_mcmc = false)
    
    nodes = Dict{Int, Vector{SPNNode}}()
    @showprogress 1 "Constructing GP nodes..." for r in gpRegions
    
        s = vec(all((X .> (r.min' - overlap)) .& (X .< (r.max' + overlap)), 2))
        xx = X[s,:]'
        yy = y[s]
        
        # construct GPs
        gp_nodes = []
        for (ki, kernel_function) in enumerate(kernelFunctions)
            
            if do_mcmc
                
                gp = GP(xx, yy, deepcopy(meanFunction), deepcopy(kernel_function), copy(noise))
                set_priors!(gp.k, kernelPriors[ki])
                
                samples = mcmc(gp; nIter=1000,burnin=0,thin=100);
                
                for i in 1:size(samples,2)
                    
                    node = GPLeaf{Any}(nextID(), 
                        GP(xx, yy, deepcopy(meanFunction), deepcopy(kernel_function), copy(noise))
                        )
                    set_params!(node.gp, samples[:,i])
                    update_target!(node.gp)
                    node.parents = SPNNode[]
                    node.minx = r.min
                    node.maxx = r.max
                    push!(gp_nodes, node)
                end
            else
               
                node = GPLeaf{Any}(nextID(), 
                    GP(xx, yy, deepcopy(meanFunction), deepcopy(kernel_function), copy(noise))
                    )
                node.parents = SPNNode[]
                node.minx = r.min
                node.maxx = r.max
                push!(gp_nodes, node)
            end
        end
        nodes[RegionIDs[r]] = gp_nodes
    end
    
    return buildNodes(rootRegion, RegionIDs, PartitionIDS, nodes, rootRegion)[1]
end

@everywhere function buildNodes(r::NDSumRegion, RegionIDs, PartitionIDS, nodes::Dict, rootRegion::NDSumRegion)
    if !haskey(nodes, RegionIDs[r])
        
        childrn = reduce(vcat, map(p -> buildNodes(p, RegionIDs, PartitionIDS, nodes, rootRegion), r.partitions))

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

                node.prior_weights[:] = rand(Dirichlet(ones(length(node)))) # use Dirichlet instead ?
                fill!(node.prior_weights, 1. / length(node))
                fill!(node.posterior_weights, 1. / length(node))
                
                @assert length(node) == length(node.posterior_weights)
                push!(n, node)
            end
            nodes[RegionIDs[r]] = n
        end
    end
    
    return nodes[RegionIDs[r]]
end

@everywhere function buildNodes(p::NDSplitPartition, RegionIDs, PartitionIDS, nodes::Dict, rootRegion::NDSumRegion)

    childrn = map(r -> buildNodes(r, RegionIDs, PartitionIDS, nodes, rootRegion), p.regions)
    
    n = SPNNode[]
    for ch1 in childrn[1]
        for ch2 in childrn[2]
            # construct node
            split = ones(p.dimensions) * -Inf
            split[p.dimension] = p.split
            node = FiniteSplitNode(nextID(), split)
        
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

@everywhere function buildNodes(r::NDGPRegion, RegionIDs, PartitionIDS, nodes::Dict, rootRegion::NDSumRegion)
    return nodes[RegionIDs[r]]
end
    
    