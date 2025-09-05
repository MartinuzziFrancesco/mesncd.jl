function smape(y_true, y_pred)
    @assert size(y_true) == size(y_pred) "Arrays must be the same shape"

    numerator = abs.(y_true .- y_pred)
    denominator = abs.(y_true) .+ abs.(y_pred) .+ eps()  # avoid division by zero
    smape_value = mean(numerator ./ denominator) * 200.0

    return smape_value
end

function rmse_scalar(y_true::Number, y_pred::Number)
    return abs(y_true - y_pred)
end

function compare_corr_dim(ts_data, ps_data)
    true_system = StateSpaceSet(ts_data[1, :], ts_data[2, :], ts_data[3, :])
    predicted_system = StateSpaceSet(ps_data[1, :], ps_data[2, :], ps_data[3, :])
    ts_cd = grassberger_proccacia_dim(true_system)
    ps_cd = grassberger_proccacia_dim(predicted_system)
    return rmse_scalar(ts_cd, ps_cd)
end
