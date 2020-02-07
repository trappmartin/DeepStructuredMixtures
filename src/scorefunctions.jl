export mse, sse
export mae, sae
export nlpd

# squared error, mean squared error and standard error of mean squared error
@inline se(y_true, y_pred) = (y_true - y_pred).^2
@inline mse(y_true, y_pred) = mean(se(y_true, y_pred))
@inline sse(y_true, y_pred) = std(se(y_true, y_pred)) / sqrt(size(y_true,1))

# absolute error, mean absolute error and standard error of mean absolute error
@inline ae(y_true, y_pred) = abs.(y_true - y_pred)
@inline mae(y_true, y_pred) = mean(ae(y_true, y_pred))
@inline sae(y_true, y_pred) = std(ae(y_true, y_pred)) / sqrt(size(y_true,1))

# negative log predictive density
@inline nlpd(y_true::AbstractVector, μ::AbstractVector, σ²::AbstractVector) = -mapreduce(i -> logpdf(Normal(μ[i], sqrt(σ²[i])), y_true[i]), +, 1:length(y_true))/length(y_true)
