import requests
import tarfile
import os
import tqdm

# --- Configuration for TUM freiburg1_xyz dataset ---
DATASET_NAME = "rgbd_dataset_freiburg1_xyz"
DOWNLOAD_URL = f"https://cvg.cit.tum.de/rgbd/dataset/freiburg1/{DATASET_NAME}.tgz"
BASE_DIR = 'tum_data'
FILE_NAME = f"{DATASET_NAME}.tgz"

def download_file(url, filename, chunk_size=8192):
    """Downloads a file with a progress bar."""
    if os.path.exists(filename):
        print(f"Skipping download: {filename} already exists.")
        return

    print(f"--- Starting download for {DATASET_NAME} ---")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 

        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file:
            with tqdm.tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = file.write(chunk)
                    bar.update(size)
        print(f"Downloaded {filename}.")

    except requests.exceptions.RequestException as e:
        print(f"Error: Could not download {url}. Check URL or network connection: {e}")
        return False
    return True

def extract_tgz(tgz_file, dest_dir):
    """Extracts a .tgz file and cleans up the archive."""
    if not os.path.exists(tgz_file):
        print(f"Error: Archive file {tgz_file} not found for extraction.")
        return

    print(f"--- Extracting {tgz_file} to {dest_dir} ---")
    try:
        with tarfile.open(tgz_file, 'r:gz') as tar:
            tar.extractall(path=dest_dir)
        print("Extraction complete.")
    except tarfile.TarError as e:
        print(f"Error: Could not extract {tgz_file}: {e}.")
    finally:
        # Cleanup the tgz file
        if os.path.exists(tgz_file):
            os.remove(tgz_file)
            print(f"Cleanup: Removed {tgz_file}.")


if __name__ == '__main__':
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # Download file to current directory
    download_file(DOWNLOAD_URL, FILE_NAME)
    
    # Extract file into the tum_data subdirectory
    extract_tgz(FILE_NAME, BASE_DIR)
        
    print("\n------------------------------------------------------")
    print(f"Data Setup Complete. Data is ready at '{BASE_DIR}/{DATASET_NAME}'.")
    print("Next: Run 'python3 run_vo_experiment.py'")
    print("------------------------------------------------------")