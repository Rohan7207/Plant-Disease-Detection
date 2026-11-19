
from torch.utils.data import DataLoader
from dataset import PlantDiseaseDataset

for split in ["train", "val", "test"]:
    dataset = PlantDiseaseDataset(split)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    images, labels = next(iter(loader))

    print(f"\n{split.upper()}")
    print("Total images:", len(dataset))
    print("Classes:", len(dataset.classes))
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)

    for label in labels:
        print("Class:", dataset.idx_to_class[label.item()])
