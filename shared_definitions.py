"""
Definición de modelos, DataModule y utilidades para clasificación de imágenes
con PyTorch Lightning.

Las arquitecturas ResNet-50, Inception-v4 y YOLO v11 heredan de
`BaseImageClassifier`, garantizando uniformidad en métricas, optimizador y
logging; cada subclase implementa únicamente sus diferencias.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tbparse import SummaryReader
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

# Hugging Face
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
)

# Métricas torchmetrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassSpecificity,
    MulticlassRecall,
    MulticlassConfusionMatrix,
)

# YOLO v11 (ultralytics)
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------
torch.set_float32_matmul_precision("medium")  # equilibrio precisión-rendimiento

# ---------------------------------------------------------------------------
# Etiquetas fijas
# ---------------------------------------------------------------------------
CLASS_LABELS: Tuple[str, ...] = ("colon_aca", "colon_n", "lung_aca", "lung_n")
ID2LABEL: Dict[int, str] = {i: lbl for i, lbl in enumerate(CLASS_LABELS)}
LABEL2ID: Dict[str, int] = {v: k for k, v in ID2LABEL.items()}
NUM_CLASSES: int = len(CLASS_LABELS)

# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class LC25000DataModule(pl.LightningDataModule):
    """Carga y preprocesa el dataset LC25000 para 4 clases."""

    IMG_SIZE = 224
    NORMALIZE = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def __init__(
        self,
        data_dir: str | Path = "lung_colon_image_set",
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

        self.train_tfm = transforms.Compose(
            [
                transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                self.NORMALIZE,
            ]
        )
        self.test_tfm = transforms.Compose(
            [
                transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
                transforms.ToTensor(),
                self.NORMALIZE,
            ]
        )

    # ---------------- Lightning hooks -----------------------------------
    def prepare_data(self):
        for subset in ("Train and Validation Set", "Test Set"):
            if not (self.data_dir / subset).is_dir():
                raise FileNotFoundError(f"{subset!r} no está en {self.data_dir}")

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            full_train = ImageFolder(
                self.data_dir / "Train and Validation Set", transform=self.train_tfm
            )
            idx_tr, idx_val = self._stratified_split(full_train)
            self.ds_train = Subset(full_train, idx_tr)
            self.ds_val = Subset(
                ImageFolder(self.data_dir / "Train and Validation Set", transform=self.test_tfm),
                idx_val,
            )
        if stage in (None, "test"):
            self.ds_test = ImageFolder(self.data_dir / "Test Set", transform=self.test_tfm)

    # ---------------- Dataloaders ---------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.ds_train, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

    # ---------------- Auxiliar ------------------------------------------
    def _stratified_split(self, dataset: ImageFolder):
        targets = dataset.targets
        idx = list(range(len(dataset)))
        return train_test_split(idx, test_size=self.val_split, stratify=targets, random_state=self.seed)


# ---------------------------------------------------------------------------
# Base Classifier
# ---------------------------------------------------------------------------


class BaseImageClassifier(pl.LightningModule):
    """Métricas, optimizador y logging unificados."""

    id2label = ID2LABEL

    def __init__(self, lr: float = 1e-4, weight_decay: float = 1e-5, scheduler_patience: int = 5):
        super().__init__()

        # métricas
        mk_acc = lambda: MulticlassAccuracy(NUM_CLASSES)
        mk_f1 = lambda: MulticlassF1Score(NUM_CLASSES)
        mk_prec = lambda: MulticlassPrecision(NUM_CLASSES)
        mk_spec = lambda: MulticlassSpecificity(NUM_CLASSES)
        mk_rec = lambda: MulticlassRecall(NUM_CLASSES)
        mk_cm = lambda: MulticlassConfusionMatrix(NUM_CLASSES)

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc, self.val_acc, self.test_acc = mk_acc(), mk_acc(), mk_acc()
        self.train_f1, self.val_f1, self.test_f1 = mk_f1(), mk_f1(), mk_f1()
        self.train_prec, self.val_prec, self.test_prec = mk_prec(), mk_prec(), mk_prec()
        self.train_spec, self.val_spec, self.test_spec = mk_spec(), mk_spec(), mk_spec()
        self.train_rec, self.val_rec, self.test_rec = mk_rec(), mk_rec(), mk_rec()
        self.val_cm, self.test_cm = mk_cm(), mk_cm()

        self.save_hyperparameters()

    # ---------------- Forward (to be implemented) -----------------------
    def forward(self, x):  # type: ignore[override]
        raise NotImplementedError

    # ---------------- Shared step ---------------------------------------
    def _shared_step(self, batch, stage: str):
        imgs, labels = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(1)

        # actualizar métricas
        getattr(self, f"{stage}_acc")(preds, labels)
        getattr(self, f"{stage}_f1")(preds, labels)
        getattr(self, f"{stage}_prec")(preds, labels)
        getattr(self, f"{stage}_spec")(preds, labels)
        getattr(self, f"{stage}_rec")(preds, labels)
        if stage in ("val", "test"):
            getattr(self, f"{stage}_cm")(preds, labels)

        # log solo por época
        log_kw = dict(on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{stage}_loss", loss, prog_bar=True, **log_kw)
        for m in ("acc", "f1", "prec", "spec", "rec"):
            self.log(f"{stage}_{m}", getattr(self, f"{stage}_{m}"), **log_kw)
        return loss

    # Lightning hooks
    def training_step(self, b, _): return self._shared_step(b, "train")
    def validation_step(self, b, _): return self._shared_step(b, "val")
    def test_step(self, b, _): return self._shared_step(b, "test")

    # ---------------- Optimizer ----------------------------------------
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", patience=self.hparams.scheduler_patience, factor=0.5
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_f1"}}


# ---------------------------------------------------------------------------
# Specific models
# ---------------------------------------------------------------------------


class ResNet50Classifier(BaseImageClassifier):
    """Fine-tuning de *microsoft/resnet-50*."""

    def __init__(self, model_name="microsoft/resnet-50", freeze_backbone=True, **kw):
        super().__init__(**kw)
        cfg = AutoConfig.from_pretrained(
            model_name, num_labels=NUM_CLASSES, id2label=ID2LABEL, label2id=LABEL2ID
        )
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name, config=cfg, ignore_mismatched_sizes=True
        )
        if freeze_backbone:
            for p in self.model.parameters(): p.requires_grad = False
            for p in self.model.classifier.parameters(): p.requires_grad = True

    def forward(self, x): return self.model(pixel_values=x).logits


class InceptionV4Classifier(BaseImageClassifier):
    """Fine-tuning de *timm/inception_v4.tf_in1k*."""

    def __init__(self, model_name="timm/inception_v4.tf_in1k", freeze_backbone=True, **kw):
        super().__init__(**kw)
        cfg = AutoConfig.from_pretrained(
            model_name, num_labels=NUM_CLASSES, id2label=ID2LABEL, label2id=LABEL2ID
        )
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name, config=cfg, ignore_mismatched_sizes=True
        )
        if freeze_backbone:
            for p in self.model.parameters(): p.requires_grad = False
            for p in self.model.timm_model.last_linear.parameters(): p.requires_grad = True

    def forward(self, x): return self.model(pixel_values=x).logits


class YOLOv11Classifier(BaseImageClassifier):
    """Fine-tuning ligero de *yolo11n-cls.pt*."""

    def __init__(self, weights="yolo11n-cls.pt", freeze_backbone=True, **kw):
        super().__init__(**kw)
        base = YOLO(weights)
        self.model = base.model
        last = self.model.model[-1]
        last.linear = nn.Linear(last.linear.in_features, NUM_CLASSES)
        if freeze_backbone:
            for n, p in self.model.named_parameters():
                if ".linear" not in n: p.requires_grad = False

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, (tuple, list)) else out


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------


def create_trainer(logger_name: str, filename: str):
    logger = TensorBoardLogger("tb_logs", name=logger_name)
    cbs = [
        RichProgressBar(refresh_rate=20),
        EarlyStopping("val_f1", patience=3, mode="max"),
        ModelCheckpoint(monitor="val_f1", mode="max", filename=filename, save_top_k=1),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    return Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=cbs,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )


# ---------------------------------------------------------------------------
# Quick inference
# ---------------------------------------------------------------------------


def classify_image(model: pl.LightningModule, img_path: str | Path, dm: LC25000DataModule):
    """Predice y muestra la etiqueta + probabilidades de una imagen."""
    model.eval().to(model.device)
    img = Image.open(img_path).convert("RGB")
    tensor = dm.test_tfm(img).unsqueeze(0).to(model.device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), 1)[0]
    conf, idx = torch.max(probs, 0)
    lbl = ID2LABEL[idx.item()]

    plt.imshow(img); plt.axis("off")
    plt.title(f"{lbl} ({conf.item():.2%})"); plt.show()
    for i, p in enumerate(probs): print(f"{ID2LABEL[i]:<12}: {p.item():.4f}")


# ---------------------------------------------------------------------------
# Post-training analysis
# ---------------------------------------------------------------------------


def draw_curves(log_dir: str | Path):
    """Grafica curvas de loss / accuracy / F1 si existen en TensorBoard."""
    df = SummaryReader(log_dir).scalars
    wanted = {
        "train_loss": "Train Loss",
        "val_loss": "Validation Loss",
        "train_acc": "Train Accuracy",
        "val_acc": "Validation Accuracy",
        "train_f1": "Train F1",
        "val_f1": "Validation F1",
    }
    present = {k: v for k, v in wanted.items() if k in df["tag"].unique()}
    if not present:
        raise ValueError("No se encontraron tags compatibles en " + str(log_dir))

    df_steps = df[df["tag"].isin(present.keys())]
    df_epochs = df[df["tag"] == "epoch"].groupby("value").first().reset_index()
    df_pivot = df_steps.pivot_table(index="step", columns="tag", values="value").rename(columns=present)

    max_labels = 12
    stride = max(1, math.ceil(len(df_epochs) / max_labels))
    xticks = df_epochs["step"][::stride].to_list()
    xlabels = df_epochs["value"].astype(int)[::stride].to_list()

    sns.set_style("whitegrid")
    metrics_groups = [
        ("Loss", ["Train Loss", "Validation Loss"]),
        ("Accuracy", ["Train Accuracy", "Validation Accuracy"]),
        ("F1-Score", ["Train F1", "Validation F1"]),
    ]
    nplots = sum(bool(set(cols) & set(df_pivot.columns)) for _, cols in metrics_groups)
    fig, axes = plt.subplots(1, nplots, figsize=(6 * nplots, 4), sharex=True)

    if nplots == 1: axes = [axes]
    i = 0
    for title, cols in metrics_groups:
        cols = [c for c in cols if c in df_pivot.columns]
        if not cols: continue
        sns.lineplot(ax=axes[i], data=df_pivot[cols], marker="o")
        axes[i].set(title=title, ylabel=title)
        axes[i].set_xticks(xticks); axes[i].set_xticklabels(xlabels, rotation=45, ha="right")
        i += 1
    plt.tight_layout(); plt.show()


def get_all_preds_and_labels(model: pl.LightningModule, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Inferencia"):
            logits = model(imgs.to(model.device))
            preds.append(torch.argmax(logits, 1).cpu()); labels.append(lbls.cpu())
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def print_confusion_matrix(model: pl.LightningModule, dm: LC25000DataModule):
    y_true, y_pred = get_all_preds_and_labels(model, dm.test_dataloader())
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.title("Matriz de Confusión (Test)"); plt.ylabel("Real"); plt.xlabel("Predicha")
    plt.xticks(rotation=45); plt.yticks(rotation=0); plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
__all__ = [
    "LC25000DataModule",
    "ResNet50Classifier",
    "InceptionV4Classifier",
    "YOLOv11Classifier",
    "create_trainer",
    "classify_image",
    "draw_curves",
    "print_confusion_matrix",
    "get_all_preds_and_labels",
]
