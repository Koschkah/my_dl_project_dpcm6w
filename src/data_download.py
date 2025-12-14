from pathlib import Path
import re

def extract_file_id(url: str) -> str:
    m = re.search(r"/file/d/([^/]+)/", url)
    if not m:
        raise ValueError(f"Nem találok file_id-t ebben az URL-ben: {url}")
    return m.group(1)

def download_gdrive_if_missing(url: str, out_path: Path):
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[download] OK, már megvan: {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    import gdown 

    file_id = extract_file_id(url)
    dl_url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[download] Letöltés: {out_path.name} ...")

    # fuzzy=True kezeli a Drive-os “nem közvetlen” linkeket is
    gdown.download(url=dl_url, output=str(out_path), quiet=False, fuzzy=True)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"[download] Sikertelen letöltés: {out_path}")

# --- használat ---
DATA_DIR = Path("/data")  # vagy ahova szeretnéd

files = [
    ("https://drive.google.com/file/d/1raXsWCCw0Hvi3d64o2-LYPKHAuNfO4ZX/view?usp=sharing", DATA_DIR / "stop_times.txt"),
    ("https://drive.google.com/file/d/1wXawryRcuLnKy3GDWrUEeO6g0VHfNAfd/view?usp=drive_link", DATA_DIR / "vehicle_updates.csv"),
    ("https://drive.google.com/file/d/19TFp8Hk200DB5sWtOxGu82klrTAzYOhZ/view?usp=sharing", DATA_DIR / "trips.txt"),
]

for url, path in files:
    download_gdrive_if_missing(url, path)
print("[download] Minden fájl letöltve.")