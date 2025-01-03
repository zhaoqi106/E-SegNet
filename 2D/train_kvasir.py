import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch import loggers
from lightning.pytorch.strategies import DDPStrategy

from datasets.dataset_KvasirSeg import KvasirSEGDataset
from network_module import Net
from model.E_SegNet_2D import E_SegNet_2D

L.seed_everything(42, workers=True)

torch.set_float32_matmul_precision("medium")

@hydra.main(config_path="configs", config_name="kvasirseg", version_base=None)
def main(cfg):
    logger = loggers.TensorBoardLogger("logs/", name=str(cfg.run_name))
    model = E_SegNet_2D(model_name=cfg.model_name, image_size=cfg.img_size, num_classes=cfg.num_classes)
    dataset = KvasirSEGDataset(batch_size=cfg.batch_size, img_size=cfg.img_size)

    net = Net(
        model=model,
        criterion=instantiate(cfg.criterion),
        optimizer=cfg.optimizer,
        lr=cfg.lr,
        scheduler=cfg.scheduler,
    )
    trainer = instantiate(cfg.trainer, logger=logger, strategy=DDPStrategy(find_unused_parameters=True))
    trainer.fit(net, dataset)
    trainer.test(net, dataset)


if __name__ == "__main__":
    main()
