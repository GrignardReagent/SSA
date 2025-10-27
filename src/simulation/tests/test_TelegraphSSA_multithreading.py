import os
from pathlib import Path

os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ["JULIA_NUM_THREADS"] = "2"  # or set to desired number of threads
import pytest
from juliacall import Main as jl


SIMULATION_DIR = Path("/home/ianyang/stochastic_simulations/julia/simulation")
JULIA_PROJECT_DIR = SIMULATION_DIR.parent
TELEGRAPHSSA_PATH = SIMULATION_DIR / "TelegraphSSA.jl"


@pytest.fixture(scope="module")
def julia_env():
    jl.seval(
        f'using Pkg; '
        f'Pkg.activate("{JULIA_PROJECT_DIR.as_posix()}"); '
        f'Pkg.instantiate()',
    )
    jl.seval("using DataFrames, NPZ")
    jl.include(str(TELEGRAPHSSA_PATH))
    jl.seval("using .TelegraphSSA")
    return jl


def _require_multiple_threads(julia_env):
    nthreads = int(julia_env.seval("Threads.nthreads()"))
    if nthreads < 2:
        pytest.skip("Julia runtime provides a single thread; multithreading test skipped.")
    return nthreads


def test_simulate_telegraph_model_uses_multiple_threads(julia_env):
    nthreads = _require_multiple_threads(julia_env)
    num_parameter_sets = max(4, nthreads * 2)
    parameter_sets_expr = f"""
        [Dict("sigma_b" => 1.0, "sigma_u" => 1.0, "rho" => 10.0,
              "d" => 1.0, "label" => i) for i in 0:{num_parameter_sets - 1}]
    """
    julia_env.parameter_sets = julia_env.seval(parameter_sets_expr)
    julia_env.time_points = julia_env.seval("collect(range(0.0, stop=1.0, length=3))")

    setup_instrumentation = """
        import Base.Threads: SpinLock, threadid, lock, unlock

        thread_log_lock = SpinLock()
        thread_ids = Set{Int}()

        function record_thread!()
            tid = threadid()
            lock(thread_log_lock)
            push!(thread_ids, tid)
            unlock(thread_log_lock)
            nothing
        end

        Base.@eval TelegraphSSA begin
            function run_single(telegraph_model, u0, tspan, params, time_points, label)
                Main.record_thread!()
                jinput = JumpInputs(telegraph_model, u0, tspan, params)
                jprob  = JumpProblem(jinput; save_positions=(false, false))
                sol       = solve(jprob, SSAStepper(), saveat = time_points)
                m_counts  = Array(sol)[3, :]
                return [label; m_counts...]
            end

            function run_ensemble(telegraph_model, u0, tspan, params, time_points, label, traj)
                Main.record_thread!()
                jinput = JumpInputs(telegraph_model, u0, tspan, params)
                jprob  = JumpProblem(jinput; save_positions=(false, false))
                eprob = EnsembleProblem(jprob)
                esol = solve(eprob, SSAStepper(); trajectories=traj, saveat=time_points)

                ensemble_array = Array(esol)
                m_counts_all = ensemble_array[3, :, :]

                rows = Vector{Vector{Any}}(undef, traj)
                for i in 1:traj
                    m_counts = m_counts_all[:, i]
                    rows[i] = [label; m_counts...]
                end

                return rows
            end
        end
    """

    teardown_instrumentation = """
        Base.@eval TelegraphSSA begin
            function run_single(telegraph_model, u0, tspan, params, time_points, label)
                jinput = JumpInputs(telegraph_model, u0, tspan, params)
                jprob  = JumpProblem(jinput; save_positions=(false, false))
                sol       = solve(jprob, SSAStepper(), saveat = time_points)
                m_counts  = Array(sol)[3, :]
                return [label; m_counts...]
            end

            function run_ensemble(telegraph_model, u0, tspan, params, time_points, label, traj)
                jinput = JumpInputs(telegraph_model, u0, tspan, params)
                jprob  = JumpProblem(jinput; save_positions=(false, false))
                eprob = EnsembleProblem(jprob)
                esol = solve(eprob, SSAStepper(); trajectories=traj, saveat=time_points)

                ensemble_array = Array(esol)
                m_counts_all = ensemble_array[3, :, :]

                rows = Vector{Vector{Any}}(undef, traj)
                for i in 1:traj
                    m_counts = m_counts_all[:, i]
                    rows[i] = [label; m_counts...]
                end

                return rows
            end
        end

        empty!(thread_ids)
    """

    julia_env.seval(setup_instrumentation)

    captured_ids = None
    try:
        julia_env.seval("empty!(thread_ids)")
        for _ in range(3):
            julia_env.seval("simulate_telegraph_model(parameter_sets, time_points, 1)")
        captured_ids = julia_env.seval("collect(thread_ids)")
    finally:
        julia_env.seval(teardown_instrumentation)

    python_ids = {int(tid) for tid in captured_ids}
    assert len(python_ids) >= 2, f"Expected multiple Julia threads, captured {python_ids}"
    assert all(1 <= tid <= nthreads for tid in python_ids)
