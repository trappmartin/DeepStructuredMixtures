mutable struct GPSumNode <: SumNode{Any}
    id::Int
    parents::Vector{SPNNode}
    children::Vector{SPNNode}
    prior_weights::Vector{Float64}
    posterior_weights::Vector{Float64}
    Z::Float64
    zDirty::Bool
        
    posterior::Float64
    posteriorDirty::Bool
    
    function GPSumNode(id, split; parents = SPNNode[])
        new(id, parents, SPNNode[], Float64[], Float64[], 0., true, 0., true)
    end
end

mutable struct FiniteSplitNode <: ProductNode
    id::Int
    parents::Vector{SPNNode}
    children::Vector{SPNNode}
    split::Vector{Float64}

    posterior::Float64
    posteriorDirty::Bool

    function FiniteSplitNode(id, split; parents = SPNNode[])
        new(id, parents, SPNNode[], split, 0., true)
    end
end

mutable struct GPLeaf{T} <: Leaf{Any}
    id::Int
    gp::GaussianProcesses.GPE
    parents::Vector{SPNNode}
    
    function GPLeaf{T}(id, gp) where T <: Any
        new(id, gp)
    end
end