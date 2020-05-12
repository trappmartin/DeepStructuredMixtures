using ProgressMeter
export train!

function train!(model::Union{DSMGP,PoE,gPoE,rBCM}, optim; iterations = 10_000, λ = 0.05, randinit = true, earlystop = 10)
    train!(model.root, model.D, model.gpmap, optim, iterations=iterations, λ=λ, randinit = randinit, earlystop = earlystop)
end

function train!(spn::Union{GPSumNode,GPSplitNode}, D::AbstractMatrix, gpmap::BiDict, optim;
                iterations = 10_000,
                λ = 0.05, # early stopping
                sharedGradients = false,
                randinit = true, earlystop = 10
                )

    gp = leftGP(spn)

    n = gp isa Array ? sum(map(sum, nparams.(gp))) : sum(nparams(gp))
    hyp = randinit ? randn(n) : reduce(vcat, params(gp, logscale=true))
    grad = zeros(n)

    nodes = SumProductNetworks.getOrderedNodes(spn)
    ids = map(n -> n.id, nodes)
    L = AxisArray(zeros(length(ids)), ids)

    p = Progress(iterations, 1, "Training...")

    c = 0
    δ = 0.0
    ℓ = zeros(iterations)

    P = SDiagonal()
    K = SDiagonal()
    if sharedGradients && (gp isa GaussianProcess)
        P = distancematrix(spn, gp.kernel, getx(spn))
        K = copyvec(P,1)
        fill!(K, 0.0)
        kernelmatrix!(gp.kernel, K, P)
    end

    for iteration in 1:iterations

        # set the parameter
        setparams!(spn, hyp)

        # fit model
        fit!(spn, D, gpmap)

        # compute mll
        fill!(L, 0.0)
        mll!(spn, L)

        ℓ[iteration] = L[spn.id]
        δ = iteration > 10 ? abs(ℓ[(iteration)] - mean(ℓ[(iteration-9):(iteration-1)])) : Inf
        ProgressMeter.next!(p; showvalues = [(:iter,iteration), (:delta,δ), (:c,c), (:llh,L[spn.id])])

        # early stopping
        if δ < λ
            c += 1
        else
            c = 0
        end

        if c >= earlystop
            @info "Early stopping at iteration $iteration with δ: $δ"
            return spn, ℓ[1:iteration]
        end

        if sharedGradients && (gp isa GaussianProcess)
            fill!(K, 0.0)
            kernelmatrix!(gp.kernel, K, P)
            updategradients!(spn, K, P, D, gpmap)
        else
            updategradients!(spn)
        end

        fill!(grad, 0.0)
        ∇mll!(spn, 0.0, 0.0, L, L[spn.id], grad)
        Flux.Optimise.apply!(optim, hyp, grad)
        hyp += grad
    end

    setparams!(spn, hyp)
    fit!(spn, D, gpmap)

    @info "Exit after $iterations iterations with δ: $δ"
    return spn, ℓ
end

function train!(gp::GaussianProcess;
                iterations = 10_000,
                optim = RMSProp(),
                λ = 0.1 # early stopping
               )

    n = sum(nparams(gp))
    hyp = randn(n)
    oldhyp = hyp
    grad = zeros(n)

    δ = 0.0
    ℓ = zeros(iterations)
    p = Progress(iterations, 1, "Training...")

    for iteration in 1:iterations

        # set the parameter
        setparams!(gp, hyp)

        # fit model
        update_cholesky!(gp)

        # compute mll
        ℓ[iteration] = mll(gp)

        if isnan(ℓ[iteration])
            setparams!(gp, oldhyp)
            update_cholesky!(gp)
            return gp, ℓ[1:iteration]
        end

        δ = iteration > 10 ? abs(ℓ[(iteration)] - mean(ℓ[(iteration-9):(iteration-1)])) : Inf
        ProgressMeter.next!(p; showvalues = [(:iter,iteration), (:delta,δ), (:llh,ℓ[iteration])])

        # early stopping
        if δ < λ
            @info "Early stopping at iteration $iteration with δ: $δ"
            return gp, ℓ[1:iteration]
        end

        updategradients!(gp)

        fill!(grad, 0.0)
        ∇mll!(gp, grad)

        oldhyp = copy(hyp)
        Flux.Optimise.apply!(optim, hyp, grad)
        hyp += grad
    end

    setparams!(gp, hyp)
    update_cholesky!(gp)

    @info "Exit after $iterations iterations with δ: $δ"
    return gp, ℓ
end
