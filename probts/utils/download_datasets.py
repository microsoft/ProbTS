from pathlib import Path
from gluonts.dataset.repository.datasets import get_dataset, dataset_names
from tqdm import tqdm


if __name__ == '__main__':
    save_path = Path('./datasets')
    for dataset in tqdm(dataset_names):
        print(f"generate {dataset}")
        ds = get_dataset(dataset, path=save_path, regenerate=False)
        print(ds.metadata)
        print(len(ds.train))