@everywhere abstract type AbstractRegion end;

mutable struct SplitPartition

    split::Int

#    nodes::Vector{FiniteSplitNode}
    regions::Vector{AbstractRegion}
end

mutable struct SumRegion <: AbstractRegion
    min::Int
    max::Int

#    nodes::Vector{GPSumNode}
    partitions::Vector{SplitPartition}
end

mutable struct GPRegion <: AbstractRegion
    min::Int
    max::Int

#    nodes::Vector{GPLeaf}
end

Base.:(==)(x::SumRegion, y::SumRegion) = (x.min == y.min) & (x.max == y.max)
Base.:(==)(x::GPRegion, y::GPRegion) = (x.min == y.min) & (x.max == y.max)
Base.:(==)(x::SumRegion, y::GPRegion) = (x.min == y.min) & (x.max == y.max)
Base.:(==)(x::GPRegion, y::SumRegion) = (x.min == y.min) & (x.max == y.max)

function poonDomingos(δ, minX, maxX)
        
    rootRegion = SumRegion(minX, maxX, SplitPartition[])

    toProcess = AbstractRegion[rootRegion]
    processed = AbstractRegion[rootRegion]

    while !isempty(toProcess)

        r = pop!(toProcess)

        # get partitions on the x-axis
        (splits) = if (r.max - r.min) > δ

            (parts, prem) = divrem((r.max - r.min), δ)

            splits = δ * collect(1:parts) + r.min

            if splits[end] == r.max
                splits = splits[1:end-1]
            end

            splits
        else
            []
        end

        # create partitions and new regions
        for split in splits

            region1 = if (split - r.min) > δ # construct sum region, otherwise gp region
                SumRegion(r.min, split, SplitPartition[])
            else
                GPRegion(r.min, split)
            end

            region2 = if (r.max - split) > δ # construct sum region, otherwise gp region
                SumRegion(split, r.max, SplitPartition[])
            else
                GPRegion(split, r.max)
            end

            r1 = if !(region1 in processed)
                if isa(region1, SumRegion)
                    push!(toProcess, region1)
                end
                push!(processed, region1)
                region1
            else
                processed[findfirst(region1 .== processed)]
            end

            r2 = if !(region2 in processed)
                if isa(region2, SumRegion)
                    push!(toProcess, region2)
                end
                push!(processed, region2)
                region2
            else
                processed[findfirst(region2 .== processed)]
            end

            p = SplitPartition(split, [r1, r2])
            push!(r.partitions, p)
        end

    end

    allRegions = processed # All regions
    sumRegions = filter(r -> isa(r, SumRegion), allRegions) # Regions with sum nodes
    gpRegions = filter(r -> isa(r, GPRegion), allRegions) # Regions with GP nodes

    allPartitions = SplitPartition[]
    for r in sumRegions
        push!(allPartitions, r.partitions...)
    end
    allPartitions = unique(allPartitions); # All partitions

    return (rootRegion, sumRegions, gpRegions, allPartitions)
end

@everywhere mutable struct NDSplitPartition
    split::Float64
    dimension::Int
    dimensions::Int
    regions::Vector{AbstractRegion}
end

@everywhere mutable struct NDSumRegion <: AbstractRegion
    min::Vector{Float64}
    max::Vector{Float64}
    partitions::Vector{NDSplitPartition}
end

@everywhere mutable struct NDGPRegion <: AbstractRegion
    min::Vector{Float64}
    max::Vector{Float64}
end

Base.:(==)(x::NDSumRegion, y::NDSumRegion) = all((x.min .== y.min) .& (x.max .== y.max))
Base.:(==)(x::NDGPRegion, y::NDGPRegion) = all((x.min .== y.min) .& (x.max .== y.max))
Base.:(==)(x::NDSumRegion, y::NDGPRegion) = all((x.min .== y.min) .& (x.max .== y.max))
Base.:(==)(x::NDGPRegion, y::NDSumRegion) = all((x.min .== y.min) .& (x.max .== y.max))

function poonDomingos_ND(δ::Vector, minX::Vector, maxX::Vector, maxDepth::Int, minSamples::Int, X)
        
    rootRegion = NDSumRegion(minX, maxX, SplitPartition[])

    toProcess = AbstractRegion[rootRegion]
    regionDepth = Dict{AbstractRegion, Int}(rootRegion => 0)
    processed = AbstractRegion[rootRegion]
    selectedDimensions = Dict{Int, Int}()
    
    D = length(minX)
    @assert length(minX) == length(maxX)
    
    while !isempty(toProcess)

        r = pop!(toProcess)
        @assert isa(r, NDSumRegion)
        
        d = if haskey(selectedDimensions, regionDepth[r])
            selectedDimensions[regionDepth[r]]
        else
            # draw a dimension at random
            rand(1:D)
        end
        
        selectedDimensions[regionDepth[r]] = d
                
        # check if we should continue building the graph
        depth_not_exceeded = regionDepth[r] < (maxDepth - 1)
        
        
        splits = []
        
        c = 0
        while isempty(splits)
            
            d = if haskey(selectedDimensions, regionDepth[r])
                selectedDimensions[regionDepth[r]]
            else
                # draw a dimension at random
                rand(1:D)
            end
        
            # get partitions on the x-axis
            splits = if (r.max[d] - r.min[d]) > δ[d]

                (parts, prem) = divrem((r.max[d] - r.min[d]), δ[d])

                splits_ = δ[d] * collect(1:parts) + r.min[d]

                if splits_[end] == r.max[d]
                    splits_ = splits_[1:end-1]
                end

                splits_
            else
                []
            end
            c += 1
            
            @assert c < 500 "$(r.max .- r.min) -> $(δ), depth: $(regionDepth[r]), $((r.max[d] - r.min[d]) > δ[d]), d: $(d), $(length(processed))"
        end
        
        # create partitions and new regions
        for split in splits

            create_sum_region = (r.max .- r.min) .> δ
            create_sum_region[d] = (split - r.min[d]) > δ[d]
            
            rmax = copy(r.max)
            rmax[d] = split
                        
            s = sum(all((X .> r.min') .& (X .< rmax'), 2))
                                    
            enough_samples = s >= minSamples
            
            region1 = if any(create_sum_region) & depth_not_exceeded & enough_samples # construct sum region, otherwise gp region
                NDSumRegion(r.min, rmax, SplitPartition[])
            else
                NDGPRegion(r.min, rmax)
            end
            
            if isa(region1, NDGPRegion)
                println("Samples in expert: ", s)
            end
            
            create_sum_region = (r.max .- r.min) .> δ
            create_sum_region[d] = (r.max[d] - split) > δ[d]
            
            rmin = copy(r.min)
            rmin[d] = split
            
            s = sum(all((X .> rmin') .& (X .< r.max'), 2))
            
            enough_samples = s >= minSamples

            region2 = if any(create_sum_region) & depth_not_exceeded & enough_samples # construct sum region, otherwise gp region
                NDSumRegion(rmin, r.max, SplitPartition[])
            else
                NDGPRegion(rmin, r.max)
            end
            
                        
            if isa(region2, NDGPRegion)
                println("Samples in expert: ", s)
            end
                        
            if !depth_not_exceeded
                @assert isa(region1, NDGPRegion)
                @assert isa(region2, NDGPRegion)
            end
            
            r1 = if !(region1 in processed)
                if isa(region1, NDSumRegion)
                    push!(toProcess, region1)
                end
                push!(processed, region1)
                region1
            else
                processed[findfirst(region1 .== processed)]
            end

            r2 = if !(region2 in processed)
                if isa(region2, NDSumRegion)
                    push!(toProcess, region2)
                end
                push!(processed, region2)
                region2
            else
                processed[findfirst(region2 .== processed)]
            end
            
            if (rmax[d] - r.min[d]) <= 2.5*δ[d]
                regionDepth[r1] = regionDepth[r] + 1
            else
                println("remaining size: ", (r.max[d] - rmin[d]), " > ", δ[d])
                regionDepth[r1] = regionDepth[r]
            end
            
            if (r.max[d] - rmin[d]) <= 2.5*δ[d]
                regionDepth[r2] = regionDepth[r] + 1
            else
                println("remaining size: ", (r.max[d] - rmin[d]), " > ", δ[d])
                regionDepth[r2] = regionDepth[r]
            end
            
            p = NDSplitPartition(split, d, D, [r1, r2])
            push!(r.partitions, p)
        end
        

    end

    allRegions = processed # All regions
    sumRegions = filter(r -> isa(r, NDSumRegion), allRegions) # Regions with sum nodes
    gpRegions = filter(r -> isa(r, NDGPRegion), allRegions) # Regions with GP nodes

    allPartitions = NDSplitPartition[]
    for r in sumRegions
        push!(allPartitions, r.partitions...)
    end
    allPartitions = unique(allPartitions); # All partitions

    return (rootRegion, sumRegions, gpRegions, allPartitions)
end