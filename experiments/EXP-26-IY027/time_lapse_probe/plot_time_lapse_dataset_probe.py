from pathlib import Path
from urllib.parse import quote, urljoin

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import requests
import scipy.io as sio
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
DRYAD_BASE_URL = "https://datadryad.org"
DOI = "10.5061/dryad.nvx0k6dzz"
TARGET_FILE = "210610_FOV1.mat"
OUT_DIR = SCRIPT_DIR / "time_lapse_probe"
FIGURE_NAME = "IY027_shimizu_fret_time_series_probe.png"
REQUEST_TIMEOUT = 60


def dryad_url(path: str) -> str:
    """Return an absolute Dryad URL from an API-relative or absolute link."""
    return urljoin(DRYAD_BASE_URL, path)


def get_json(session: requests.Session, url: str) -> dict:
    """Fetch JSON and fail with the response body if Dryad returns an error."""
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def get_latest_version_files(session: requests.Session) -> list[dict]:
    """Return all files listed for the latest Dryad dataset version."""
    dataset_url = dryad_url(f"/api/v2/datasets/{quote('doi:' + DOI, safe='')}")
    dataset = get_json(session, dataset_url)
    version = get_json(session, dryad_url(dataset["_links"]["stash:version"]["href"]))

    files_url = dryad_url(version["_links"]["stash:files"]["href"])
    files = []
    while files_url:
        page = get_json(session, files_url)
        files.extend(page.get("_embedded", {}).get("stash:files", []))
        files_url = page.get("_links", {}).get("next", {}).get("href")
        files_url = dryad_url(files_url) if files_url else None

    return files


def download_dryad_file(session: requests.Session, file_info: dict, mat_path: Path) -> None:
    """Download a Dryad file, guarding against API auth errors saved as data."""
    download_urls = [dryad_url(file_info["_links"]["stash:download"]["href"])]

    file_id = file_info["_links"]["self"]["href"].rstrip("/").split("/")[-1]
    download_urls.append(dryad_url(f"/downloads/file_stream/{file_id}"))

    errors = []
    for url in download_urls:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        content_type = response.headers.get("content-type", "")
        if response.ok and "json" not in content_type and "html" not in content_type:
            mat_path.write_bytes(response.content)
            return
        errors.append(f"{url} returned {response.status_code} ({content_type})")

    raise RuntimeError(
        f"Could not download {TARGET_FILE} from Dryad. "
        "If Dryad requires browser-mediated download, place the file at "
        f"{mat_path} and rerun. Tried: {'; '.join(errors)}"
    )


def ensure_mat_file() -> Path:
    """Download the target `.mat` file unless it is already cached locally."""
    OUT_DIR.mkdir(exist_ok=True)
    mat_path = OUT_DIR / TARGET_FILE
    if mat_path.exists() and mat_path.stat().st_size > 0:
        return mat_path

    with requests.Session() as session:
        files = get_latest_version_files(session)
        file_info = next((x for x in files if x["path"] == TARGET_FILE), None)
        if file_info is None:
            available = ", ".join(x["path"] for x in files[:10])
            raise FileNotFoundError(
                f"{TARGET_FILE} was not listed in Dryad dataset {DOI}. "
                f"First listed files: {available}"
            )
        download_dryad_file(session, file_info, mat_path)

    return mat_path


def load_first_cell_response(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the first cell's time and smoothed normalized FRET response matrices."""
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    reorg_data = mat["reorgData"]
    resp_data = np.ravel(reorg_data.resp_data)
    if resp_data.size == 0:
        raise ValueError(f"No resp_data entries found in {mat_path}")

    first_cell = resp_data[0]
    time_s = np.atleast_2d(np.asarray(first_cell.t, dtype=float))
    smooth_activity = np.atleast_2d(np.asarray(first_cell.smooth_a, dtype=float))
    if time_s.shape != smooth_activity.shape:
        raise ValueError(
            "Time and smoothed activity arrays have different shapes: "
            f"{time_s.shape} != {smooth_activity.shape}"
        )

    return time_s, smooth_activity


def plot_responses(time_s: np.ndarray, smooth_activity: np.ndarray) -> Path:
    """Plot aligned normalized FRET responses from one cell."""
    fig_path = OUT_DIR / FIGURE_NAME
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    palette = sns.color_palette("colorblind", n_colors=3)
    relative_time_s = time_s - time_s[:, [0]]
    common_time_s = np.nanmedian(relative_time_s, axis=0)

    mean_activity = np.nanmean(smooth_activity, axis=0)
    sem_activity = np.nanstd(smooth_activity, axis=0, ddof=1) / np.sqrt(
        smooth_activity.shape[0]
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.3, 1.4]},
    )
    ax_trace, ax_heatmap = axes

    individual_lines = ax_trace.plot(
        common_time_s,
        smooth_activity.T,
        color="0.45",
        linewidth=0.9,
        alpha=0.28,
        zorder=1,
    )
    individual_lines[0].set_label("Individual smooth responses")
    ax_trace.fill_between(
        common_time_s,
        mean_activity - sem_activity,
        mean_activity + sem_activity,
        color=palette[0],
        alpha=0.18,
        linewidth=0,
        label="Mean +/- SEM",
        zorder=2,
    )
    ax_trace.plot(
        common_time_s,
        mean_activity,
        color=palette[0],
        linewidth=2.4,
        label="Mean smooth response",
        zorder=3,
    )
    ax_trace.axhline(0, color="0.2", linewidth=0.8, alpha=0.5)
    ax_trace.set_ylabel("Normalized FRET activity / a.u.")
    ax_trace.set_title(
        f"Cell 1 FRET responses aligned to response start (n={smooth_activity.shape[0]})"
    )
    ax_trace.grid(True, axis="y", color="0.9", linewidth=0.8)
    ax_trace.legend(loc="upper right", frameon=True, framealpha=0.9)

    heatmap = ax_heatmap.imshow(
        smooth_activity,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        extent=[
            common_time_s[0],
            common_time_s[-1],
            smooth_activity.shape[0] + 0.5,
            0.5,
        ],
    )
    ax_heatmap.set_xlabel("Time after response start / s")
    ax_heatmap.set_ylabel("Response index")
    ax_heatmap.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
    colorbar = fig.colorbar(
        heatmap,
        ax=ax_heatmap,
        location="right",
        pad=0.01,
        fraction=0.035,
    )
    colorbar.set_label("Smoothed activity / a.u.")
    colorbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))

    for ax in axes:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def main() -> None:
    mat_path = ensure_mat_file()
    time_s, smooth_activity = load_first_cell_response(mat_path)
    fig_path = plot_responses(time_s, smooth_activity)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
