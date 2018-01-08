function add!(parent::GPSumNode, child::SPNNode)
    if !(child in parent.children)
        push!(parent.children, child)
        push!(parent.prior_weights, 1.)
        push!(parent.posterior_weights, 1.)
        push!(child.parents, parent)
        
        parent.prior_weights ./= sum(parent.prior_weights)
        parent.posterior_weights ./= sum(parent.posterior_weights)
    end
    
    @assert sum(parent.prior_weights) ≈ 1. "Weights should sum up to one, sum(w) = $(sum(parent.prior_weights))"
    @assert sum(parent.posterior_weights) ≈ 1. "Weights should sum up to one, sum(w) = $(sum(parent.prior_weights))"
 end

function add!(parent::FiniteSplitNode, child::SPNNode)
     if !(child in parent.children)
         push!(parent.children, child)
         push!(child.parents, parent)
     end
end

Base.show(io::IO, n::GPLeaf) = 
    print(io, "Gaussian Process Leaf Node [ID: ", n.id, ", LLH: ", round(n.gp.target, 3), "]")

Base.show(io::IO, n::FiniteSplitNode) = 
    print(io, "Split (Product) Node [ID: ", n.id, ", split: ", round.(n.split, 3), "]")

Base.show(io::IO, n::GPSumNode) = 
    print(io, "Gaussian Process Sum Node [ID: ", n.id, ", \n\t w_prior: ", 
        round.(n.prior_weights, 3), ", \n\t w_posterior: ", 
        round.(n.posterior_weights, 3), "]")

function isDirty(n::GPSumNode)
    return n.posteriorDirty
end