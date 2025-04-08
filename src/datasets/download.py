from huggingface_hub import hf_hub_download, list_repo_files
import argparse
from joblib import Parallel, delayed
import tarfile
import os
import shutil
import random
import time

def download_file_from_huggingface(repo_id, filename, local_dir, overwrite=False, keep_tar=False):
    """download a file from huggingface and extract the file."""
    file_non_tar = local_dir + "/" + filename.replace('.tar', '')
    if os.path.exists(file_non_tar) and not overwrite:
        # if file exists do not download again (file is different from tar file)
        return
    try:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=local_dir)
        # extract the tar ball
        if tarfile.is_tarfile(file_path):
            with tarfile.open(file_path, "r") as tar:
                tar.extractall(os.path.dirname(file_path))
            if not keep_tar:
                os.remove(file_path)
    except Exception as e:
        # if something goes wrong, make sure to remove the file, so it can be downloaded again
        if os.path.exists(file_non_tar):
            shutil.rmtree(file_non_tar)
        print(str(e))
        return filename

def download_data_from_huggingface(modalities=None, geoids=None, local_dir='data/uscc/', repo_id='torchgeo/CropClimateX', nr_workers=2):
    """download data from huggingface datasets."""

    if modalities and not isinstance(modalities, (list,tuple)):
        modalities = [modalities]

    if geoids and not isinstance(geoids, (list,tuple)):
        geoids = [geoids]

    files = list_repo_files(repo_id, repo_type="dataset")
    # files not in modalities (meta data)
    files_meta = [f for f in files if f.endswith(".geojson")]
    files_meta += ["minicubes_geometry.tar"]

    if not files:
        raise ValueError(f"No files found in the repository. Please check if the repository is valid: {repo_id}")

    if modalities:
        files = [f for f in files if any([m == os.path.dirname(f) for m in modalities])]
    if geoids:
        files = [f for f in files if f.endswith(".json") or f.endswith(".csv") or any([g in f for g in geoids])]
    if not files:
        raise ValueError(f"No files found in the repository for modalities: {modalities} and geoids: {geoids}")

    # shuffle files
    random.shuffle(files)

    out_meta = Parallel(n_jobs=1)(delayed(download_file_from_huggingface)(repo_id=repo_id, filename=f, local_dir=local_dir) for f in files_meta)
    os.makedirs(local_dir, exist_ok=True)
    out = Parallel(n_jobs=nr_workers)(delayed(download_file_from_huggingface)(repo_id=repo_id, filename=f, local_dir=local_dir) for f in files)
    out = [o for o in out if o is not None]
    out_meta = [o for o in out_meta if o is not None]
    out.extend(out_meta)
    if len(out) > 0:
        print(f"\nFailed to download/extract files: {out}.\nSometimes Hugging Face rejects a download request, for example if there are too many requests. Please try again later.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download dataset. Specify modalities, and geoids.')
    parser.add_argument('-m', '--modalities', nargs='+', type=str, help='List of modalities to download')
    parser.add_argument('-ids', '--geoids', nargs='+', type=str, help='List of counties (geoids) to download data for')
    parser.add_argument('--local_dir', type=str, help='Output folder to save downloaded data')
    parser.add_argument('--nr_workers', type=int, help='Number of workers to use for parallel download')

    args = parser.parse_args()
    args = dict(vars(args))
    print(f"Downloading with args: {args}")

    download_data_from_huggingface(**args)
    # example usage: python src/datasets/download.py -m dem modis -ids 26059 --local_dir data/CropClimateX --nr_workers 8
    # or use the function: download_data_from_huggingface(['dem'], ['55083'], local_dir='data/CropClimateX')
