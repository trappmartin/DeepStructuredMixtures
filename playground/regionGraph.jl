abstract type AbstractRegion end;

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