import numpy as np
from datasets import load_dataset


def _load_data(batch_size: int, split: str, length:int = 1000):
    ds = load_dataset("ylecun/mnist", split=split)
    ds = ds.select(range(length))

    for batch in ds.iter(batch_size=batch_size):
        images = np.array([np.array(image).reshape(-1) for image in batch["image"]])
        images = images.astype(np.float32) / 255.0

        labels = np.array(batch["label"], dtype=np.int64)

        yield images, labels

def get_train_dataset(batch_size: int):
    return _load_data(batch_size, split="train")

def get_test_dataset(batch_size: int):
    return _load_data(batch_size, split="test", length=20)

if __name__ == "__main__":
    ds = get_train_dataset(10)
    images, labels = next(ds)
    print(images.shape)
    print(labels.shape)
