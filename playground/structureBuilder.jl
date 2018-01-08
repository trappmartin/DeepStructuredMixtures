struct SPNParameters
    K # number of children under a sum node
    numSamples # minimum percentage of samples per part
    minSamples # minimum number of samples to further construct an SPN
    kernelFunction # shared kernel function
    meanFunction # shared mean function
    noise # shared noise parameter
end

function makeSplitNode(x, y, parameter, depth, maximumDepth)
    
    N = length(y)
    
    xmin = minimum(x)
    xmax = maximum(x) - xmin
    
    # make random (1D) split
    split = (rand() * xmax) + xmin
    n = min(sum(x .<= split), sum(x .> split))
    c = 0
    
    while n < (N * parameter.numSamples)
        @assert c < 200 "Could not find a split"
        split = (rand() * xmax) + xmin
        n = min(sum(x .<= split), sum(x .> split))
        c += 1
    end
    
    node = FiniteSplitNode(nextID(), Float64[split])
    
    s = x .<= split
    
    if (N <= parameter.minSamples) | (depth >= maximumDepth)
        # append two GP leaves
        leaf1 = GPLeaf{Any}(nextID(),
            GP(reshape(x[s], 1, sum(s)), y[s], 
                parameter.meanFunction, parameter.kernelFunction, parameter.noise))
        leaf1.parents = SPNNode[]
    
        leaf2 = GPLeaf{Any}(nextID(),
            GP(reshape(x[.!s], 1, sum(.!s)), y[.!s], 
                parameter.meanFunction, parameter.kernelFunction, parameter.noise))
        leaf2.parents = SPNNode[]
    
        add!(node, leaf1)
        add!(node, leaf2)
    else
        # append two sum nodes
        child1 = makeSumNode(x[s], y[s], parameter, depth + 1, maximumDepth)
        child2 = makeSumNode(x[.!s], y[.!s], parameter, depth + 1, maximumDepth)
        
        add!(node, child1)
        add!(node, child2)
    end
    
    return node
end

function makeSumNode(x, y, parameter, depth, maximumDepth)
    
    node = GPSumNode(nextID(), Int[]);
    
    newChildren = pmap(k -> makeSplitNode(x, y, parameter, depth, maximumDepth), 1:parameter.K)
    
    for child in newChildren
        add!(node, child)
    end
    
    fill!(node.prior_weights, 1. / parameter.K)
    fill!(node.posterior_weights, 1. / parameter.K)
    
    return node
end