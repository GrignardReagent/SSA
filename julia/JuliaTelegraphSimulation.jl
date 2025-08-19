module JuliaTelegraphSimulation

using Catalyst
# using DiffEqJump          # for JumpProblem, Direct(), SSAStepper
using OrdinaryDiffEqDefault      # not strictly needed for SSA, but fine to keep
using DataFrames
using Random
using Base.Threads

export simulate_telegraph_model

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
function simulate_telegraph_model(parameter_sets, time_points, ntrajectories; num_cores::Int=nthreads())
    # Define the reaction network once
    rn = @reaction_network begin
        sigma_u, G → Gs          # gene activation
        sigma_b, Gs → G          # gene deactivation
        rho,     G → G + M       # mRNA production
        d,       M → 0           # mRNA degradation
    end

    # initial molecule counts
    u0 = [:G => 1, :Gs => 0, :M => 0]
    tspan = (time_points[1], time_points[end])

    rows = Vector{Vector{Any}}()  # Any to allow String or Int labels

    for ps in parameter_sets
        # parameters in the order rn.ps (Catalyst guarantees param order in rn.ps)
        params = (; sigma_u = ps["sigma_u"],
                   sigma_b = ps["sigma_b"],
                   rho     = ps["rho"],
                   d       = ps["d"])

        label = get(ps, "label", 0)

        # Preallocate for this parameter set
        local_rows = Vector{Vector{Any}}(undef, ntrajectories)

        if num_cores <= 1
            for i in 1:ntrajectories
                local_rows[i] = run_single(rn, u0, tspan, params, time_points, label)
            end
        else
            @threads for i in 1:ntrajectories
                local_rows[i] = run_single(rn, u0, tspan, params, time_points, label)
            end
        end

        append!(rows, local_rows)
    end

    colsyms = [:label; Symbol.("time_" .* string.(time_points))]
    return DataFrame(rows, colsyms)
end

# One SSA trajectory, returns a row: [label, M(t1), M(t2), ...]
function run_single(rn, u0, tspan, params, time_points, label)
    # SSA over a discrete problem is the canonical pattern
    dprob     = DiscreteProblem(rn, u0, tspan, params)
    jprob     = JumpProblem(rn, dprob, Direct())
    sol       = solve(jprob, SSAStepper(), saveat = time_points)
    m_counts  = Array(sol)[3, :]  # species order is [G, Gs, M]
    return Any[label; m_counts...]  # label + mRNA counts
end

end # module
