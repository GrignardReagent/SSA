module TelegraphSSA

using Catalyst
using JumpProcesses
using DifferentialEquations
using DataFrames
using Random
using Base.Threads

export simulate_telegraph_model # this can be used directly when the module is imported

"""
    simulate_telegraph_model(parameter_sets, time_points, ntrajectories)

Simulate mRNA counts from the 2-state telegraph model via Gillespie SSA.

- `parameter_sets`: a Vector of Dicts with keys:
    "sigma_u", "sigma_b", "rho", "d", and optional "label" (string or int).
- `time_points`: Vector of times at which to save M counts (e.g. `collect(0.0:1.0:100.0)`).
- `ntrajectories`: number of trajectories per parameter set.

Parallelism is two-level:
- Outer: `@threads` distributes parameter sets across Julia threads.
- Inner: Each parameter set's trajectories use `EnsembleSerial` (serial within
  @threads to avoid over-subscription).

Thread count is controlled by `JULIA_NUM_THREADS`, which must be set in the
environment *before* Julia starts.

Returns a `DataFrame` with columns: `:label, :time_t1, :time_t2, ...`.
"""
function simulate_telegraph_model(parameter_sets, time_points, traj::Int)
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

    # Pre-allocate with fixed size so each thread writes to a unique slice —
    # avoids the data race that push!/append! on a shared Vector would cause.
    n_params = length(parameter_sets)
    rows = Vector{Vector{Any}}(undef, n_params * traj)

    # Convert Python parameter dicts to Julia dicts BEFORE threading
    # (accessing Python objects from multiple threads causes segfaults)
    julia_params = [
        (
            sigma_u=ps["sigma_u"],
            sigma_b=ps["sigma_b"],
            rho=ps["rho"],
            d=ps["d"],
            label=get(ps, "label", 0)
        )
        for ps in parameter_sets
    ]

    @threads for i in 1:n_params
        ps = julia_params[i]
        # parameters dict for the reaction network
        params = Dict(
            :sigma_u => ps.sigma_u,
            :sigma_b => ps.sigma_b,
            :rho     => ps.rho,
            :d       => ps.d,
        )

        label = ps.label
        start_idx = (i - 1) * traj + 1

        if traj == 1
            # single traj
            rows[start_idx] = run_single(telegraph_model, u0, tspan, params, time_points, label)
        else
            # multiple trajs, use ensemble
            ensemble_results = run_ensemble(telegraph_model, u0, tspan, params, time_points, label, traj)
            for j in 1:traj
                rows[start_idx + j - 1] = ensemble_results[j]
            end
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
    # Use EnsembleSerial when called from @threads to avoid nested parallelism issues.
    # EnsembleThreads() is still beneficial when there's a single parameter set.
    esol = solve(eprob, SSAStepper(), EnsembleSerial(); trajectories=traj, saveat=time_points)
    
    # Extract M counts for each trajectory
    ensemble_array = Array(esol)  # shape: (species, time_points, trajectories)
    m_counts_all = ensemble_array[3, :, :]  # M is the 3rd species
    
    # Create one row per trajectory
    rows = Vector{Vector{Any}}(undef, traj)
    for i in 1:traj
        m_counts = m_counts_all[:, i]  # M counts for trajectory i, we are only interested in M so that's the only thing we extract
        rows[i] = [label; m_counts...]
    end
    
    return rows
end

end # module
