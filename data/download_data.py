"""
Download METR-LA and PEMS-BAY datasets from Google Drive.
Original source: https://github.com/liyaguang/DCRNN
"""
import os
import zipfile
import urllib.request

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Google Drive direct download links (DCRNN official mirrors)
METR_LA_URL = "https://drive.google.com/uc?export=download&id=1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC"
PEMS_BAY_URL = "https://drive.google.com/uc?export=download&id=1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq"


def download_gdrive(url: str, dest_path: str) -> None:
    """Download file from Google Drive with session cookie workaround."""
    import requests

    session = requests.Session()
    response = session.get(url, stream=True)
    # Handle large-file confirmation token
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token:
        response = session.get(url, params={"confirm": token}, stream=True)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print(f"  Saved -> {dest_path}")


def prepare_metr_la():
    """Download and extract METR-LA dataset."""
    out_dir = os.path.join(DATA_DIR, "metr-la")
    os.makedirs(out_dir, exist_ok=True)
    # Files expected: metr_la.h5, adj_mx.pkl
    # We'll pull from a reliable Zenodo / academic mirror
    files = {
        "metr_la.h5": "https://zenodo.org/record/5724362/files/metr_la.h5",
        "adj_mx.pkl": "https://zenodo.org/record/5724362/files/adj_mx.pkl",
    }
    for fname, url in files.items():
        dest = os.path.join(out_dir, fname)
        if os.path.exists(dest):
            print(f"  Already exists: {fname}")
            continue
        print(f"  Downloading {fname} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"  [WARN] Could not download {fname}: {e}")
            print(f"  Please manually download from: https://github.com/liyaguang/DCRNN")


def prepare_pems_bay():
    """Download and extract PEMS-BAY dataset."""
    out_dir = os.path.join(DATA_DIR, "pems-bay")
    os.makedirs(out_dir, exist_ok=True)
    files = {
        "pems_bay.h5": "https://zenodo.org/record/5724362/files/pems_bay.h5",
        "adj_mx_bay.pkl": "https://zenodo.org/record/5724362/files/adj_mx_bay.pkl",
    }
    for fname, url in files.items():
        dest = os.path.join(out_dir, fname)
        if os.path.exists(dest):
            print(f"  Already exists: {fname}")
            continue
        print(f"  Downloading {fname} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"  [WARN] Could not download {fname}: {e}")
            print(f"  Please manually download from: https://github.com/liyaguang/DCRNN")


if __name__ == "__main__":
    print("=== Downloading METR-LA ===")
    prepare_metr_la()
    print("=== Downloading PEMS-BAY ===")
    prepare_pems_bay()
    print("Done. Run data/preprocess.py next.")
