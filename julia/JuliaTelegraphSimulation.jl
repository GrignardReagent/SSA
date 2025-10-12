module JuliaTelegraphSimulation

using Catalyst
using JumpProcesses
using DifferentialEquations
using DataFrames
using Random
using Base.Threads

export simulate_telegraph_model # this can be used directly when the module is imported

"""
    simulate_telegraph_model(parameter_sets, time_points, ntrajectories; num_cores=nthreads())

Simulate mRNA counts from the 2-state telegraph model via Gillespie SSA.

- `parameter_sets`: a Vector of Dicts with keys:
    "sigma_u", "sigma_b", "rho", "d", and optional "label".
- `time_points`: Vector of times at which to save M counts (e.g. `collect(0.0:1.0:100.0)`).
- `ntrajectories`: number of trajectories per parameter set.
- `num_cores`: number of threads to use (defaults to `Threads.nthreads()`).

Returns a `DataFrame` with columns: `:label, :time_t1, :time_t2, ...`.
"""
function simulate_telegraph_model(parameter_sets, time_points, traj::Int; num_cores::Int=nthreads())
    # the network
    telegraph_model = @reaction_network begin
        sigma_u, G --> G_star # gene activation
        sigma_b, G_star --> G # gene deactivation
        rho,     G --> G + M  # mRNA production
        d,       M --> 0      # mRNA degradation
    end

    # initial molecule counts
    u0 = [:G => 1, :G_star => 0, :M => 0]
    tspan = (time_points[1], time_points[end])

    rows = Vector{Vector{Any}}()  # Any to allow String or Int labels

    @threads for ps in parameter_sets
        # parameters in the order rn.ps (Catalyst guarantees param order in rn.ps)
        params = Dict(
            :sigma_u => ps["sigma_u"],
            :sigma_b => ps["sigma_b"],
            :rho     => ps["rho"],
            :d       => ps["d"],
        )

        label = Int32(get(ps, "label", 0))

        if traj == 1
            # single traj
            row = run_single(telegraph_model, u0, tspan, params, time_points, label)
            push!(rows, row)
        else
            # multiple trajs, use ensemble
            ensemble_results = run_ensemble(telegraph_model, u0, tspan, params, time_points, label, traj)
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

# One SSA trajectory, returns a row: [label, M(t1), M(t2), ...]
function run_single(telegraph_model, u0, tspan, params, time_points, label)
    jinput = JumpInputs(telegraph_model, u0, tspan, params)
    jprob  = JumpProblem(jinput; save_positions=(false, false)) # Disable saving at jump positions
    sol       = solve(jprob, SSAStepper(), saveat = time_points)
    m_counts  = Array(sol)[3, :]  # species order is [G, Gs, M], so we choose the 3rd row
    return [label; m_counts...]  # label + mRNA counts
end

# run an ensemble of trajectories
function run_ensemble(telegraph_model, u0, tspan, params, time_points, label, traj)
    jinput = JumpInputs(telegraph_model, u0, tspan, params)
    jprob  = JumpProblem(jinput; save_positions=(false, false))
    eprob = EnsembleProblem(jprob)
    esol = solve(eprob, SSAStepper(); trajectories=traj, saveat=time_points)
    
    # Extract M counts for each trajectory
    ensemble_array = Array(esol)  # shape: (species, time_points, trajectories)
    m_counts_all = ensemble_array[3, :, :]  # M is the 3rd species
    
    # Create one row per trajectory
    rows = Vector{Vector{Any}}(undef, traj)
    for i in 1:traj
        m_counts = m_counts_all[:, i]  # M counts for trajectory i
        rows[i] = [label; m_counts...]
    end
    
    return rows
end

end # module
