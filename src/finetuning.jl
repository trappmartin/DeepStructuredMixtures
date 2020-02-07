export finetune!

function finetune!(model::DSMGP; iterations = 1000, optim = RMSProp(), λ = 0.5)
    return finetune!(model.root, model.D, model.gpmap, iterations=iterations, optim=optim, λ=λ)
end

function finetune!(spn::Union{GPSumNode,GPSplitNode}, D::AbstractMatrix, gpmap;
                iterations = 1000,
                optim = RMSProp(),
                λ = 0.5 # early stopping
                )

    gp = leftGP(spn)

    n = gp isa Array ? sum(map(sum, nparams.(gp))) : sum(nparams(gp))
    grad = zeros(n)

    nodes = SumProductNetworks.getOrderedNodes(spn);
    ids = map(n -> n.id, nodes)

    gps = filter(n -> n isa GPNode, nodes);

    hyp = Dict(gp.id => reduce(vcat, params(gp.dist, logscale=true)) for gp in gps)

    L = AxisArray(zeros(length(ids)), ids)
    l = 0.0
    δ = Inf
    c = 0
    ℓ = zeros(iterations)

    Dd = copy(D)
    Dd[diagind(Dd)] .= 1.0

    p = Progress(iterations, 1, "Training...")
    for iteration in 1:iterations

        l = 0.0
        for gp in gps
            hyp_ = hyp[gp.id]

            # set the parameter
            setparams!(spn, hyp_)

            # fit model
            fit!(spn, D, gpmap)

            # compute mll
            fill!(L, 0.0)
            mll!(spn, L)

            updategradients!(spn)
            l += L[gp.id]

            fill!(grad, 0.0)
            ∇mll!(spn, 0.0, 0.0, L, L[spn.id], grad, view(D, gpmap.x[gp.id], :), gpmap)
            apply!(optim, hyp_, grad)
            hyp[gp.id] += grad
        end

        ℓ[iteration] = l

        δ = iteration > 10 ? abs(ℓ[(iteration)] - mean(ℓ[(iteration-9):(iteration-1)])) : Inf
        ProgressMeter.next!(p; showvalues = [(:iter,iteration), (:delta,δ), (:c,c), (:llh,L[spn.id])])

        # early stopping
        if δ < λ
            c += 1
        else
            c = 0
        end

        if c >= 10
            @info "Early stopping at iteration $iteration with δ: $δ"

            for gp in gps
                setparams!(gp.dist, hyp[gp.id])
                update_cholesky!(gp.dist)
            end
            return spn, ℓ
        end
    end

    for gp in gps
        setparams!(gp.dist, hyp[gp.id])
        update_cholesky!(gp.dist)
    end

    return spn, ℓ
end

