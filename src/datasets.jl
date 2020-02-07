using StatsFuns

export nonstationary

function nonstationary(n; σ²=0.4)

    # Create toy data
    x = range(-200, stop = 200, length = n);

    f1 = 3.0*sin.(-3 .+ 0.2.*x[1:Int(ceil(0.25*n))])
    f1 = vcat(f1, 0*sin.(0.1*x[Int(ceil(0.25*n))+1:Int(ceil(0.75*n))]))
    f1 = vcat(f1, 3.0*sin.(2.8 .+ 0.2.*x[Int(ceil(0.75*n)) .+ 1:end]))

    f2 = 100*normpdf.(110, 20, x) + 100*normpdf.(-10, 20, x)

    x = x .- mean(x)
    x = x / std(x)
    f1 = f1 .- mean(f1)
    f1 = f1 / std(f1)

    noise = sqrt.((σ².*exp.(f2)))
    y = f1 + noise.*randn(size(x))
    x=x[:]*10
    y=y[:];

    return reshape(x,:,1), y, noise
end
