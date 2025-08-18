module JuliaTelegraphSimulation

using Catalyst
using DifferentialEquations
using DataFrames
using Random
using Base.Threads

"""
    simulate_telegraph_model(parameter_sets, time_points, size; num_cores = nothing)

Simulate multiple telegraph-model systems using the Gillespie algorithm via
Catalyst. Each element of `parameter_sets` should be a dictionary (or
AbstractDict) with keys `"sigma_u"`, `"sigma_b"`, `"rho"`, `"d"`, and an
optional `"label"` identifying the condition.  `time_points` is a vector of
times at which to save the mRNA counts and `size` gives the number of
trajectories per parameter set.  If `num_cores` is provided and greater than
one, simulations for a given parameter set are executed in parallel across
threads.  A `DataFrame` with one row per trajectory and columns
`label`, `time_t1`, `time_t2`, ... is returned.
"""
function simulate_telegraph_model(parameter_sets, time_points, size; num_cores::Union{Int,Nothing}=nothing)
    num_cores === nothing && (num_cores = nthreads())

    # Define the reaction network once
    rn = @reaction_network begin
        σ_u, G --> Gs       # gene activation
        σ_b, Gs --> G       # gene deactivation
        ρ,   G --> G + M    # mRNA production
        d,   M --> 0        # mRNA degradation
    end

    u0 = [1, 0, 0]                                   # initial populations
    tspan = (time_points[1], time_points[end])
    all_rows = Vector{Vector{Int}}()

    for param_set in parameter_sets
        σ_u = param_set["sigma_u"]
        σ_b = param_set["sigma_b"]
        ρ   = param_set["rho"]
        d   = param_set["d"]
        label = get(param_set, "label", 0)
        p = [σ_u, σ_b, ρ, d]

        local_rows = Matrix{Int}(undef, size, length(time_points) + 1)
        if num_cores == 1
            for i in 1:size
                local_rows[i, :] = run_single(rn, u0, tspan, p, time_points, label)
            end
        else
            @threads for i in 1:size
                local_rows[i, :] = run_single(rn, u0, tspan, p, time_points, label)
            end
        end
        append!(all_rows, eachrow(local_rows))
    end

    columns = [:label; Symbol.("time_" .* string.(time_points))]
    return DataFrame(all_rows, columns)
end

# helper that performs a single trajectory simulation
function run_single(rn, u0, tspan, params, time_points, label)
    prob = DiscreteProblem(rn, u0, tspan, params)
    jump_prob = JumpProblem(rn, prob, Direct())
    sol = solve(jump_prob, SSAStepper(), saveat=time_points)
    m_counts = Array(sol)[3, :]
    return [label; m_counts]
end

end # module
