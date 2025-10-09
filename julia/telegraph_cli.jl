#!/usr/bin/env julia
using Pkg
Pkg.activate(@__DIR__); Pkg.instantiate()

using Catalyst
using JumpProcesses
using DifferentialEquations
using Plots
using NPZ
using CSV, DataFrames

# Optional: headless GR (safe on servers)
# ENV["GKSwstype"] = "100"

function main(; traj::Int=1, tstop::Float64=1000.0,
               savepng::String="telegraph_plot.png",
               savenpz::Union{Nothing,String}=nothing,
               savecsv::Union{Nothing,String}=nothing)

    # the network
    telegraph_model = @reaction_network begin
        sigma_u, G --> G_star
        sigma_b, G_star --> G
        rho,     G --> G + M
        d,       M --> 0
    end

    # initial conditions, timespan, parameters
    u0     = [:G => 1, :G_star => 0, :M => 0]
    tspan  = (0.0, tstop)
    params = [:sigma_u => 0.5, :sigma_b => 0.5, :rho => 100.0, :d => 1.0]

    # the problem 
    @time begin
        jinput = JumpInputs(telegraph_model, u0, tspan, params)
        jprob  = JumpProblem(jinput; save_positions=(false, false)) # Disable saving at jump positions

        plt = nothing
        # single traj can use JumpProblem directly
        if traj == 1
            sol = solve(jprob, saveat=1.0) # must specify saveat to save at regular intervals
            plt = plot(sol; idxs=[:G,:G_star,:M], xlabel="time", ylabel="count",
                       legend=:topright)

            if savenpz !== nothing
                NPZ.npzwrite(savenpz, Array(sol))
                println("Saved data to $(savenpz)")
            end

            if savecsv !== nothing
                state_names = [first(p) for p in u0]
                state_matrix = permutedims(Array(sol))
                sol_df = DataFrame(state_matrix, state_names)
                insertcols!(sol_df, 1, :time => sol.t)
                CSV.write(savecsv, sol_df)
                println("Saved data to $(savecsv)  (size=$(size(sol_df)))")
            end

        # need to use EnsembleProblem for multiple trajs
        else
            eprob = EnsembleProblem(jprob)
            esol  = solve(eprob, SSAStepper(); trajectories=traj, saveat=1.0)
            plt   = plot(esol; idxs=3, legend=false, xlabel="time", ylabel="M")

            if savenpz !== nothing
                NPZ.npzwrite(savenpz, Array(esol))
                println("Saved data to $(savenpz)")
            end
            
            if savecsv !== nothing
                state_names = [first(p) for p in u0]
                ensemble_array = Array(esol)
                permuted = permutedims(ensemble_array, (3, 2, 1))
                species_matrix = reshape(permuted, :, length(state_names))
                time_points = esol.u[1].t
                time_col = repeat(time_points, outer=traj)
                trajectory_ids = repeat(collect(1:traj), inner=length(time_points))
                esol_df = DataFrame(species_matrix, state_names)
                insertcols!(esol_df, 1, :time => time_col)
                insertcols!(esol_df, 1, :trajectory => trajectory_ids)
                CSV.write(savecsv, esol_df)
                println("Saved data to $(savecsv)  (size=$(size(esol_df)))")
            end
        end

        savefig(plt, savepng)
    end
    println("Saved plot to $(savepng)")
end

# Minimal CLI args: traj tstop savepng savenpz
if abspath(PROGRAM_FILE) == @__FILE__
    traj    = length(ARGS) >= 1 ? parse(Int,     ARGS[1]) : 1
    tstop   = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1000.0
    savepng = length(ARGS) >= 3 ? ARGS[3] : "telegraph_plot.png"
    savenpz = length(ARGS) >= 4 ? ARGS[4] : nothing
    savecsv = length(ARGS) >= 5 ? ARGS[5] : nothing
    main(traj=traj, tstop=tstop, savepng=savepng, savenpz=savenpz, savecsv=savecsv)
end
