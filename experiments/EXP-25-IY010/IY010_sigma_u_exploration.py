import numpy as np
import time
from multiprocessing import Process, Queue


## This script is designed to find the most appropriate sigma_u for find_parameters function

# -----------------------------------------------------------------------------
# Worker function to call find_parameters inside a separate process
# -----------------------------------------------------------------------------
def _worker_process(q, parameter_set, mu, cv, autocorr):
    """
    Calls the real find_parameters and sends result or exception via Queue.
    """
    from simulation.mean_var_autocorr_sigma_u import find_parameters
    try:
        rho, sigma_b, d = find_parameters(
            parameter_set,
            mu_target=mu,
            cv_target=cv,
            autocorr_target=autocorr
        )
        q.put(("result", (rho, sigma_b, d)))
    except Exception as e:
        q.put(("error", e))

# -----------------------------------------------------------------------------
# 1) Define your targets
# -----------------------------------------------------------------------------
cv_target = 0.13          # coefficient of variation
mu_target = 0.05          # mean
autocorr_target = 100     # autocorrelation at lag 1

# -----------------------------------------------------------------------------
# 2) Define a grid of sigma_u values to search over
# -----------------------------------------------------------------------------
sigma_u_grid = np.arange(0.05, 50.0, 0.05)  # adjust min, max, step count as needed

time_limit = 60  # seconds per attempt
results = []     # store successful results

# -----------------------------------------------------------------------------
# 3) Loop through each sigma_u and attempt to converge with timeout
# -----------------------------------------------------------------------------
for sigma_u in sigma_u_grid:
    parameter_set = {"sigma_u": float(sigma_u)}
    print(f"Trying sigma_u = {sigma_u:.3f}...")

    # Set up a multiprocessing queue and process
    q = Queue()
    p = Process(
        target=_worker_process,
        args=(q, parameter_set, mu_target, cv_target, autocorr_target)
    )
    p.start()

    # Wait up to time_limit seconds
    p.join(time_limit)
    if p.is_alive():
        # If still running after timeout, terminate
        p.terminate()
        p.join()
        print(f"❌ Timeout after {time_limit}s for sigma_u = {sigma_u:.3f}")
        continue

    # Process finished: get result from queue
    if not q.empty():
        kind, payload = q.get()
        if kind == "result":
            rho, sigma_b, d = payload
            # Note: elapsed time is approximate here
            print(f"✅ Converged: rho={rho:.4f}, sigma_b={sigma_b:.4f}, d={d:.4f}")
            results.append({
                "sigma_u": sigma_u,
                "rho": rho,
                "sigma_b": sigma_b,
                "d": d
            })
        else:
            # payload is exception
            print(f"❌ No convergence for sigma_u = {sigma_u:.3f}: {payload}")
    else:
        print(f"❌ No output received for sigma_u = {sigma_u:.3f}")

# -----------------------------------------------------------------------------
# 4) Summarize results
# -----------------------------------------------------------------------------
if results:
    print("\nAll successful sigma_u values:")
    for r in results:
        print(
            f"  • sigma_u={r['sigma_u']:.3f} → "
            f"rho={r['rho']:.4f}, sigma_b={r['sigma_b']:.4f}, d={r['d']:.4f}"
        )
else:
    print("No suitable sigma_u found in the provided grid.")