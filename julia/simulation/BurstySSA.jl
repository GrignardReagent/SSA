module BurstySSA

using Distributions: Geometric
using Catalyst
using JumpProcesses
using DifferentialEquations
using DataFrames
using Random
using Base.Threads

# symbolic registration must be evaluated at the top level 
@register_symbolic Geometric(b)

export simulate_bursty_model # this can be used directly when the module is imported

"""
    simulate_bursty_model(parameter_sets, time_points, ntrajectories; num_cores=nthreads())

Simulate mRNA counts from the 2-state bursty model via Gillespie SSA.

- `parameter_sets`: a Vector of Dicts with keys:
    "k", "b", "d", and optional "label".
- `time_points`: Vector of times at which to save M counts (e.g. `collect(0.0:1.0:100.0)`).
- `ntrajectories`: number of trajectories per parameter set.
- `num_cores`: number of threads to use (defaults to `Threads.nthreads()`).

Returns a `DataFrame` with columns: `:label, :time_t1, :time_t2, ...`.
"""
function simulate_bursty_model(parameter_sets, time_points, traj::Int; num_cores::Int=nthreads())
    
    @parameters b
    r = rand(Geometric(1/b)) + 1
    # the network
    bursty_model = @reaction_network bursty_model begin
        k, G --> G + $r*P
        d, P --> 0 
        end

    # initial molecule counts
    u0 = [:G => 1, :P => 0]
    tspan = (time_points[1], time_points[end])

    rows = Vector{Vector{Any}}()  # Any to allow String or Int labels

    @threads for ps in parameter_sets
        # parameters in the order rn.ps (Catalyst guarantees param order in rn.ps)
        params = Dict(
            :k => ps["k"],
            :b => ps["b"],
            :d => ps["d"],
        )

        label = Int32(get(ps, "label", 0))

        if traj == 1
            # single traj
            row = run_single(bursty_model, u0, tspan, params, time_points, label)
            push!(rows, row)
        else
            # multiple trajs, use ensemble
            ensemble_results = run_ensemble(bursty_model, u0, tspan, params, time_points, label, traj)
            append!(rows, ensemble_results)
        end
    end

    colsyms = [:label; Symbol.("time_" .* string.(time_points))]

    # Convert rows to a proper matrix format
    data_matrix = Matrix{Any}(undef, length(rows), length(colsyms))
    for (i, row) in enumerate(rows)
        data_matrix[i, :] = row
    end

    df = DataFrame(data_matrix, colsyms)
    return df
end

# One SSA trajectory, returns a row: [label, P(t1), P(t2), ...]
function run_single(bursty_model, u0, tspan, params, time_points, label)
    jinput = JumpInputs(bursty_model, u0, tspan, params)
    jprob  = JumpProblem(jinput; save_positions=(false, false)) # Disable saving at jump positions
    sol       = solve(jprob, SSAStepper(), saveat = time_points)
    p_counts  = Array(sol)[2, :]  # species order is [G, P], so we choose the 2nd row
    return [label; p_counts...]  # label + P counts
end

# run an ensemble of trajectories
function run_ensemble(bursty_model, u0, tspan, params, time_points, label, traj)
    jinput = JumpInputs(bursty_model, u0, tspan, params)
    jprob  = JumpProblem(jinput; save_positions=(false, false))
    eprob = EnsembleProblem(jprob)
    esol = solve(eprob, SSAStepper(); trajectories=traj, saveat=time_points)
    
    # Extract P counts for each trajectory
    ensemble_array = Array(esol)  # shape: (species, time_points, trajectories)
    p_counts_all = ensemble_array[2, :, :]  # P is the 2nd species
    
    # Create one row per trajectory
    rows = Vector{Vector{Any}}(undef, traj)
    for i in 1:traj
        p_counts = p_counts_all[:, i]  # P counts for trajectory i
        rows[i] = [label; p_counts...]
    end
    
    return rows
end

end # module