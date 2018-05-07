function nextID()
    global gID
    gID += 1
    return gID
end

function getAllSplits(spn)

    splitNodes = filter(n -> isa(n, FiniteSplitNode), SumProductNetworks.getOrderedNodes(spn))
    allSplits = Dict{Int, Vector{Vector{Float64}}}()

    for splitNode in splitNodes
        d = depth(splitNode)
        if !haskey(allSplits, d)
            allSplits[d] = Vector{Vector{Float64}}(0)
        end

        push!(allSplits[d], splitNode.split)    
    end
    
    return allSplits
end

function getSplits(spn, minDepth)

    splitNodes = filter(n -> isa(n, FiniteSplitNode), SumProductNetworks.getOrderedNodes(spn))
    allSplits = Dict{Int, Vector{Vector{Float64}}}()

    for splitNode in splitNodes
        d = depth(splitNode)
        
        if d >= minDepth
        
            if !haskey(allSplits, d)
                allSplits[d] = Vector{Vector{Float64}}(0)
            end

            push!(allSplits[d], splitNode.split)    
        end
    end
    
    return allSplits
end

function plotSplits!(plt, splits)
    depths = sort(collect(keys(splits)))
    for d in depths
        vline!(plt, [s[1] for s in splits[d]], label = "depth $(d) splits")
    end
    plt
end