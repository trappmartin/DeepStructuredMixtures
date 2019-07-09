@recipe function f(spn::Union{GPSplitNode, GPSumNode}; β=0.95, obsv=true, var=false, n=100)

    lgp = leftGP(spn)
    rgp = rightGP(spn)

    D = lgp isa AbstractArray ? size(first(lgp).x, 1) : size(lgp.x, 1)

    if D == 1
        xlims --> (minimum(lgp.x), maximum(rgp.x))
        xmin, xmax = plotattributes[:xlims]
        x = range(xmin, stop=xmax, length=100)

        y, Σ = DeepGaussianProcessExperts.predict(spn, x)
        err = GaussianProcesses.invΦ((1+β)/2)*sqrt.(Σ)

        @series begin
            seriestype := :path
            ribbon := err
            fillcolor --> :lightblue
            color --> :black
            x,y
        end
        if obsv
            @series begin
                seriestype := :scatter
                markershape := :circle
                markercolor := :black
                getx(spn)', gety(spn)
            end
        end
    elseif D == 2

        xmin = lgp isa AbstractArray ? mapreduce(gp -> minimum(gp.x[1,:]), min, lgp) : minimum(lgp.x[1,:])
        xmax = rgp isa AbstractArray ? mapreduce(gp -> maximum(gp.x[1,:]), max, rgp) : maximum(rgp.x[1,:])
        ymin = lgp isa AbstractArray ? mapreduce(gp -> minimum(gp.x[2,:]), min, lgp) : minimum(lgp.x[2,:])
        ymax = rgp isa AbstractArray ? mapreduce(gp -> maximum(gp.x[2,:]), max, rgp) : maximum(rgp.x[2,:])

        xlims --> (xmin,xmax)
        ylims --> (ymin,ymax)
        xmin, xmax = plotattributes[:xlims]
        ymin, ymax = plotattributes[:ylims]
        x = range(xmin, stop=xmax, length=n)
        y = range(ymin, stop=ymax, length=n)
        xgrid = repeat(x', n, 1)
        ygrid = repeat(y, 1, n)

        μ, Σ = predict(spn, hcat(vec(xgrid), vec(ygrid)))

        if var
            zgrid  = reshape(Σ,n,n)
        else
            zgrid  = reshape(μ,n,n)
        end
        x, y, zgrid
    end
end
