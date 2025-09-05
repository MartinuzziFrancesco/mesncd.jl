function data_from_ode(rng, prob, tspan = (0, 10000);
        abstol=1e-13, reltol=1e-13)
    initial_conditions = prob.u0 #.+ (0.001 * randn(rng, 3))
    new_prob = remake(prob; u0 = initial_conditions)
    sol = solve(new_prob, Feagin12(), tspan=tspan, abstol=abstol, reltol=reltol)
    dt = sol.t[2] - sol.t[1]
    data = Array(sol)
    data_tr = fit(ZScoreTransform, data, dims=2)
    data = StatsBase.transform(data_tr, data)
    return dt, data_tr, data
end

function data_from_ode_standard(prob, tspan = (0, 10000);
        abstol=1e-13, reltol=1e-13)
    sol = solve(prob, Feagin12(), tspan=tspan, abstol=abstol, reltol=reltol)
    dt = sol.t[2] - sol.t[1]
    data = Array(sol)
    data_tr = fit(ZScoreTransform, data, dims=2)
    data = StatsBase.transform(data_tr, data)
    return dt, data_tr, data
end

function split_data(data, shift, train_len, predict_len)
    input_data = data[:, shift:(shift+train_len-1)]
    target_data = data[:, (shift+1):(shift+train_len)]
    test = data[:, (shift+train_len):(shift+train_len+predict_len-1)]
    return input_data, target_data, test
end
