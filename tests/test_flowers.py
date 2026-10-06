import pytest
import numpy as np
import torch

from flowers.data import (
    DataConfig,
    FlowersDataset,
    eval_transforms,
    image_path,
    load_labels,
    load_official_splits,
    train_transforms,
)
from flowers.models import build_model, count_parameters, load_checkpoint, save_checkpoint, target_layer
from flowers.evaluate import confusion_matrix, per_class_prf, top_confused_pairs
from flowers.explain import GradCAM

RAW = "/home/hatch/workspace/portfolio-projects/fine-grained-image-classification/data/raw"


def test_official_splits():
    splits = load_official_splits(RAW)
    assert len(splits["train"]) == 1020
    assert len(splits["val"]) == 1020
    assert len(splits["test"]) == 6149
    # disjoint
    assert len(set(splits["train"]) & set(splits["val"])) == 0
    assert len(set(splits["train"]) & set(splits["test"])) == 0


def test_labels_range():
    labels = load_labels(RAW)
    assert labels.shape == (8189,)
    assert labels.min() == 0 and labels.max() == 101


def test_image_files_exist():
    assert image_path(RAW, 0).exists()
    assert image_path(RAW, 8188).exists()


def test_dataset_item_shape():
    splits = load_official_splits(RAW)
    ds = FlowersDataset(RAW, splits["train"][:4], transform=eval_transforms(224))
    x, y = ds[0]
    assert x.shape == (3, 224, 224)
    assert 0 <= y <= 101
    assert len(ds) == 4


def test_train_transforms_augment():
    tf = train_transforms(224)
    assert any("RandomResizedCrop" in str(t) for t in tf.transforms)


def test_model_output_dims():
    model = build_model(num_classes=102, freeze_backbone=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 102)


def test_freeze_backbone():
    model = build_model(freeze_backbone=True)
    frozen = [p for n, p in model.named_parameters() if not p.requires_grad]
    assert len(frozen) > 0
    total, trainable = count_parameters(model)
    assert trainable < total


def test_target_layer_is_conv():
    model = build_model()
    layer = target_layer(model)
    assert isinstance(layer, torch.nn.Conv2d)


def test_checkpoint_roundtrip(tmp_path):
    model = build_model(freeze_backbone=True)
    model.eval()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(model, path, extra={"epoch": 3})
    model2 = build_model(freeze_backbone=True)
    model2.eval()
    payload = load_checkpoint(model2, path, torch.device("cpu"))
    assert payload["epoch"] == 3
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model(x), model2(x))


def test_gradcam_heatmap_shape_and_range():
    torch.manual_seed(0)
    # NOTE: backbone must be unfrozen — with frozen weights the target conv
    # layer sits outside the autograd graph and no gradients flow to it.
    model = build_model(freeze_backbone=False)
    model.eval()
    cam = GradCAM(model, target_layer(model))
    try:
        x = torch.randn(1, 3, 224, 224)
        heat = cam(x, class_idx=5)
        assert heat.shape == (224, 224)
        assert heat.min() >= 0.0 and heat.max() <= 1.0
        assert heat.max() > 0  # non-degenerate on random input
    finally:
        cam.remove()


def test_confusion_matrix_toy():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1 and cm[0, 1] == 1
    prf = per_class_prf(y_true, y_pred, 3)
    assert prf["precision"].shape == (3,)
    assert abs(prf["recall"][2] - 1.0) < 1e-9


def test_top_confused_pairs_toy():
    cm = np.zeros((4, 4), dtype=np.int64)
    cm[1, 2] = 7
    cm[3, 0] = 3
    np.fill_diagonal(cm, 10)
    pairs = top_confused_pairs(cm, top_n=2)
    assert pairs[0] == (1, 2, 7)
    assert pairs[1] == (3, 0, 3)


def test_dataloader_batch(tmp_path):
    from torch.utils.data import DataLoader
    from flowers.data import DataConfig, build_dataloaders

    cfg = DataConfig(batch_size=4, num_workers=0)
    loaders = build_dataloaders(RAW, cfg, augment=False)
    x, y = next(iter(loaders["train"]))
    assert x.shape == (4, 3, 224, 224)
    assert y.shape == (4,)
