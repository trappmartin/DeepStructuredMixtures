export kernelidfunction

const invΦ = norminvcdf

function kernelidfunction(spn::Union{GPSplitNode, GPSumNode})
    lgp = leftGP(spn)
    rgp = rightGP(spn)

    xmin = lgp isa AbstractArray ? mapreduce(gp -> minimum(gp.x), min, lgp) : minimum(lgp.x)
    xmax = rgp isa AbstractArray ? mapreduce(gp -> maximum(gp.x), max, rgp) : maximum(rgp.x)

    x = range(xmin, stop=xmax, length=100)
    y = kernelid(spn, reshape(collect(x), :, 1))

    return x, y
end

@recipe function f(model::Union{DSMGP,PoE,gPoE,rBCM}; β=0.95, obsv=true, var=false, n=100, xmin=-Inf, xmax=Inf)

    root = model.root

    lgp = leftGP(root)
    rgp = rightGP(root)

    D = lgp isa AbstractArray ? first(lgp).D : lgp.D

    if D == 1

        if isinf(xmin)
            xmin = lgp isa AbstractArray ? mapreduce(gp -> minimum(gp.x), min, lgp) : minimum(lgp.x)
        end
        if isinf(xmax)
            xmax = rgp isa AbstractArray ? mapreduce(gp -> maximum(gp.x), max, rgp) : maximum(rgp.x)
        end

        xlims --> (xmin, xmax)
        xmin, xmax = plotattributes[:xlims]
        x = range(xmin, stop=xmax, length=100)

        y, Σ = predict(model, reshape(x,:,1))
        Σ[Σ .< 0] .= 0.0
        err = invΦ((1+β)/2)*sqrt.(Σ)

        @series begin
            seriestype := :path
            linewidth := 1.5
            model isa DSMGP ? label --> "DSMGP" : label --> "PoE"
            x,y
        end
        @series begin
            primary := false
            seriestype := :path
            linewidth := 1.4
            linestyle := :dot
            x,y.-err
        end
        @series begin
            primary := false
            seriestype := :path
            linewidth := 1.4
            linestyle := :dot
            x,y.+err
        end
        if obsv
            @series begin
                primary := false
                #label := "observations"
                seriestype := :scatter
                markershape := :circle
                markercolor := :black
                markersize := 0.7
                getx(root), gety(root)
            end
        end
    elseif D == 2

        xmin = lgp isa AbstractArray ? mapreduce(gp -> minimum(gp.x[:,1]), min, lgp) : minimum(lgp.x[:,1])
        xmax = rgp isa AbstractArray ? mapreduce(gp -> maximum(gp.x[:,1]), max, rgp) : maximum(rgp.x[:,1])
        ymin = lgp isa AbstractArray ? mapreduce(gp -> minimum(gp.x[:,2]), min, lgp) : minimum(lgp.x[:,2])
        ymax = rgp isa AbstractArray ? mapreduce(gp -> maximum(gp.x[:,2]), max, rgp) : maximum(rgp.x[:,2])

        xlims --> (xmin,xmax)
        ylims --> (ymin,ymax)
        xmin, xmax = plotattributes[:xlims]
        ymin, ymax = plotattributes[:ylims]
        x = range(xmin, stop=xmax, length=n)
        y = range(ymin, stop=ymax, length=n)
        xgrid = repeat(x', n, 1)
        ygrid = repeat(y, 1, n)

        μ, Σ = predict(model, hcat(vec(xgrid), vec(ygrid)))

        if var
            zgrid  = reshape(Σ,n,n)
        else
            zgrid  = reshape(μ,n,n)
        end
        x, y, zgrid
    end
end

@recipe function f(gp::GaussianProcess; β=0.95, obsv=true, var=false, n=100, xmin=-Inf, xmax=Inf)

    @assert gp.D == 1

    xmin = isinf(xmin) ? minimum(gp.x) : xmin
    xmax = isinf(xmax) ? maximum(gp.x) : xmax

    xlims --> (xmin, xmax)
    xmin, xmax = plotattributes[:xlims]
    x = range(xmin, stop=xmax, length=100)

    y, Σ = prediction(gp, reshape(x, :, 1))
    σ² = diag(Σ)
    err = invΦ((1+β)/2)*sqrt.(σ²)

    @series begin
        seriestype := :path
        ribbon := err
        fillcolor --> :lightblue
        linewidth := 1
        linestyle := :dash
        label --> "GP"
        x,y
    end
    if obsv
        @series begin
            primary := false
            #label := "observations"
            seriestype := :scatter
            markershape := :circle
            markercolor := :black
            markersize := 0.7
            gp.x, gp.y + get(gp.mean, gp.N)
        end
    end
end
