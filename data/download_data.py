"""
Download METR-LA and PEMS-BAY datasets from Hugging Face mirrors.
"""
import os
import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def download_file_with_progress(url: str, dest_path: str):
    """Download a file with streaming progress in MB."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"  Already exists ({os.path.getsize(dest_path) / 1024 / 1024:.1f} MB): {dest_path}")
        return

    print(f"  Downloading: {url}")
    print(f"  Destination: {dest_path}")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB chunks
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    print(
                        f"\r  Progress: {downloaded / 1024 / 1024:.1f} / {total_size / 1024 / 1024:.1f} MB ({percent:.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r  Downloaded: {downloaded / 1024 / 1024:.1f} MB",
                        end="",
                        flush=True,
                    )
    print("\n  Download complete!")


def prepare_metr_la():
    """Download METR-LA dataset."""
    out_dir = os.path.join(DATA_DIR, "metr-la")
    files = {
        "adj_mx.pkl": "https://huggingface.co/datasets/jimmygao3218/METRLA/resolve/main/adj_mx.pkl",
        "metr_la.h5": "https://huggingface.co/datasets/jimmygao3218/METRLA/resolve/main/metr-la.h5",
    }
    for fname, url in files.items():
        dest = os.path.join(out_dir, fname)
        download_file_with_progress(url, dest)


def prepare_pems_bay():
    """Download PEMS-BAY dataset."""
    out_dir = os.path.join(DATA_DIR, "pems-bay")
    files = {
        "adj_mx_bay.pkl": "https://huggingface.co/datasets/jimmygao3218/PEMSBAY/resolve/main/adj_mx_bay.pkl",
        "pems_bay.h5": "https://huggingface.co/datasets/jimmygao3218/PEMSBAY/resolve/main/PEMS-BAY.h5",
    }
    for fname, url in files.items():
        dest = os.path.join(out_dir, fname)
        download_file_with_progress(url, dest)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="metr-la", choices=["metr-la", "pems-bay", "both"])
    args = parser.parse_args()

    if args.dataset in ["metr-la", "both"]:
        print("\n=== Downloading METR-LA ===")
        prepare_metr_la()

    if args.dataset in ["pems-bay", "both"]:
        print("\n=== Downloading PEMS-BAY ===")
        prepare_pems_bay()

    print("\nDone! Next step: run python data/preprocess.py")
