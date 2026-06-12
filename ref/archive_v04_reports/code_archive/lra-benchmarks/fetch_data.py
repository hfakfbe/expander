import requests
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import tarfile


def download_url(url, save_path, chunk_size=1024):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(save_path, 'wb') as fd, tqdm(unit_scale=True, unit_divisor=chunk_size, total=total) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            size = fd.write(chunk)
            pbar.update(size)
    return save_path


def extract_tar(archive, dest_dir, subdir=None, mode="r:gz"):
    with tarfile.open(archive, mode) as tar:
        if subdir is None:
            tar.extractall(path=dest_dir)
        else:
            members = [tarinfo for tarinfo in tar.getmembers() if tarinfo.name.startswith(subdir)]
            tar.extractall(path=dest_dir, members=members)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", choices=["cifar10", "listops", "imdb"], help="name of dataset to download")
    parser.add_argument("--dir", type=Path, help="path to directory for saving datasets")
    args = parser.parse_args()

    datasets = {"cifar10": {"url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"},
                "listops": {"url": "https://storage.googleapis.com/long-range-arena/lra_release.gz"},
                "imdb": {"url": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"}
                }
    task = args.task
    path_dir = args.dir.resolve()
    if task == "imdb":
        path = download_url(datasets[task]["url"], path_dir / "aclImdb_v1.tar.gz")
        extract_tar(path, path_dir)
    elif task == "cifar10":
        path = download_url(datasets[task]["url"], path_dir / "cifar10.tar.gz")
        extract_tar(path, path_dir)
    elif task == "listops":
        path = download_url(datasets[task]["url"], path_dir / "lra_release.gz")
        extract_tar(path, path_dir, subdir="lra_release/listops-1000")
    else:
        assert False, f"no support for dataset named `{task}`"
