from monai.data import DataLoader, CacheDataset
from monai.data.utils import pad_list_data_collate

def get_trainloader(train_files, train_transforms):
    train_ds = CacheDataset(data=train_files, transform=train_transforms, num_workers=2)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, collate_fn=pad_list_data_collate)

    return train_loader


def get_testloader(test_files, test_transforms):
    test_ds = CacheDataset(data=test_files, transform=test_transforms, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=2, collate_fn=pad_list_data_collate)

    return test_loader