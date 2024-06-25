import torch
from ctransfer.data import Quijote
from pathlib import Path
from argparse import ArgumentParser

from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import Trainer
from lightning.pytorch import seed_everything

from ctransfer.models.resnet import ResNet
from ctransfer.models.ctransfer import cTransfer


parser = ArgumentParser()

parser.add_argument(
    "--output_dir",
    default="/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/ctransfer/models/",
    help="output_directory",
    required=False,
)
# Data hparams
parser.add_argument(
    "--baseline_root_dir",
    default="/n/holyscratch01/iaifi_lab/Lab/quijote_large/density_fields/",
    help="directory with baseline density fields",
    required=False,
)
parser.add_argument(
    "--few_shot_root_dir",
    default="/n/holyscratch01/iaifi_lab/Lab/quijote_neutrinos/density_fields/",
    help="directory with few shot density fields",
    required=False,
)
parser.add_argument(
    "--n_baseline",
    default=10_000,
    type=int,
    help="number of baseline samples used for training",
    required=False,
)
parser.add_argument(
    "--n_val",
    default=20,
    help="number of images to use for evaluation",
    required=False,
)
parser.add_argument(
    "--n_shots",
    default=80,
    type=int,
    help="number of transfer samples used for training",
    required=False,
)
parser.add_argument(
    "--n_shots_val",
    default=10,
    type=int,
    help="number of transfer samples used for validation",
    required=False,
)
parser.add_argument(
    "--n_shots_test",
    default=10,
    type=int,
    help="number of transfer samples used for testing",
    required=False,
)
parser.add_argument(
    "--resolution",
    default=256,
    type=int,
    help="Number of pixels in the density field",
    required=False,
)
parser.add_argument(
    "--cosmological_parameters",
    default=["Omega_m", "Omega_b", "h", "n_s", "sigma_8"],
    type=int,
    help="Cosmo params to fit",
    required=False,
)

parser.add_argument(
    "--few_shot_cosmological_parameters",
    default=["M_nu", "w"],
    type=int,
    help="Cosmo params to fine tune",
    required=False,
)


# ResNet hparams 
parser.add_argument(
    "--summary_dim",
    default=16,
    type=int,
    help="dimensionality of the learned summary statistic",
    required=False,
)
parser.add_argument(
    "--num_channels",
    default=(8, 16, 16, 8, 8, 8, 4, 2, 2),
    type=int,
    help="base channel of ResNet",
    required=False,
)
parser.add_argument(
    "--num_blocks",
    default=2,
    type=int,
    help="number of downsizing blocks",
    required=False,
)

# density estimator hparams 
parser.add_argument(
    "--n_transforms",
    default=5,
    type=int,
    help="number of flow transforms",
    required=False,
)

# Training hparams
parser.add_argument(
    "--learning_rate",
    default=2e-4,
    # default=5.e-5,
    help="learning rate",
    required=False,
)
parser.add_argument(
    "--grad_clip",
    default=1.0,
    help="gradient norm clipping",
    required=False,
)
parser.add_argument(
    "--total_steps",
    default=2_000,
    type=int,
    help="total training steps",
    required=False,
)
parser.add_argument(
    "--few_shot_total_steps",
    default=2_000,
    type=int,
    help="total training steps when few shot learning",
    required=False,
)

parser.add_argument(
    "--save_every",
    default=1_000,
    help="number of images to use for evaluation",
    required=False,
)
parser.add_argument(
    "--warmup",
    default=5000,
    help="learning rate warmup",
    required=False,
)
parser.add_argument(
    "--batch_size",
    default=4,
    type=int,
    help="batch size",
    required=False,
)

parser.add_argument(
    "--accumulate_gradients",
    default=None,
    type=int,
    help="steps to accumulate grads",
    required=False,
)
parser.add_argument(
    "--num_workers",
    default=2,
    type=int,
    help="workers of Dataloader",
    required=False,
)
parser.add_argument(
    "--checkpoint_every",
    default=1_000,
    type=int,
    required=False,
    help="frequency of saving checkpoints, 0 to disable during training",
)
parser.add_argument(
    "--eval_every",
    default=20,
    type=int,
    required=False,
    help="frequency of evaluating model, 0 to disable during training",
)


def setup_data(args, root_dir: str, massive_neutrinos: bool, idx_range: range, cosmological_parameters: list, shuffle: bool = False,) -> torch.utils.data.DataLoader:
    """
    Sets up the data loader for training, validation, or testing.

    Args:
        args (Namespace): The argument namespace containing various hyperparameters.
        root_dir (str): The root directory of the data.
        massive_neutrinos (bool): Whether the dataset includes massive neutrinos.
        idx_range (range): The range of indices to use for the dataset.
        cosmological_parameters (list): List of cosmological parameters to use.

    Returns:
        torch.utils.data.DataLoader: The data loader for the specified dataset.
    """
    data = Quijote(
        root=root_dir,
        idx_list=idx_range,
        resolution=args.resolution,
        redshift=0.0,
        cosmological_parameters=cosmological_parameters,
        massive_neutrinos=massive_neutrinos,
    )
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=args.num_workers,
    )
    return loader

def setup_trainer(args, total_steps: int, logger: WandbLogger, checkpoint_callback: ModelCheckpoint, early_callback: EarlyStopping) -> Trainer:
    """
    Sets up the Trainer for training the model.

    Args:
        args (Namespace): The argument namespace containing various hyperparameters.
        total_steps (int): The total number of training steps.
        logger (WandbLogger): The logger for logging metrics.
        checkpoint_callback (ModelCheckpoint): The callback for saving checkpoints.
        early_callback (EarlyStopping): The callback for early stopping.

    Returns:
        Trainer: The Trainer object configured with the specified parameters.
    """
    return Trainer(
        max_steps=total_steps,
        gradient_clip_val=args.grad_clip,
        logger=logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=None,
        val_check_interval=args.eval_every,
        accumulate_grad_batches=args.accumulate_gradients
        if args.accumulate_gradients is not None
        else 1,
        callbacks=[checkpoint_callback, early_callback],
        devices=-1,
    )


def train(args):
    """
    The main training function.

    Args:
        args (Namespace): The argument namespace containing various hyperparameters.
    """
    seed_everything(42, workers=True)
    train_loader = setup_data(
        args,
        root_dir=args.baseline_root_dir,
        massive_neutrinos=False,
        idx_range=range(args.n_baseline),
        cosmological_parameters=args.cosmological_parameters,
        shuffle=True,
    )
    val_loader = setup_data(
        args,
        root_dir=args.baseline_root_dir,
        massive_neutrinos=False,
        idx_range=range(args.n_baseline, args.n_baseline + args.n_val),
        cosmological_parameters=args.cosmological_parameters,
    )

    summarizer = ResNet(
        input_image_resolution=args.resolution,
        block_out_channels=args.num_channels,
        summary_dim=args.summary_dim,
    )

    model = cTransfer(
        summarizer=summarizer,
        n_features=len(args.cosmological_parameters),
        n_transforms=args.n_transforms,
        phase='baseline',
    )
    wandb_logger = WandbLogger(project="ctransfer", log_model=False)
    run_name = wandb_logger.experiment.name
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.output_dir) / f"{run_name}",
        save_top_k=1,
        monitor="baseline_val_loss",
        save_last=True,
        auto_insert_metric_name=True,
        every_n_train_steps=args.save_every,
    )
    early_callback = EarlyStopping(monitor="baseline_val_loss", mode="min")
    if args.n_baseline > 0:
        trainer = setup_trainer(
            args, args.total_steps, wandb_logger, checkpoint_callback, early_callback
        )
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

    # ****** Use representation in a downstream task ****** #
    few_shot_train_loader = setup_data(
        args,
        root_dir=args.few_shot_root_dir,
        massive_neutrinos=True,
        idx_range=range(args.n_shots),
        cosmological_parameters=args.cosmological_parameters
        + args.few_shot_cosmological_parameters,
        shuffle=True,
    )
    few_shot_val_loader = setup_data(
        args,
        root_dir=args.few_shot_root_dir,
        massive_neutrinos=True,
        idx_range=range(args.n_shots, args.n_shots + args.n_shots_val),
        cosmological_parameters=args.cosmological_parameters
        + args.few_shot_cosmological_parameters,
    )
    few_shot_test_loader = setup_data(
        args,
        root_dir=args.few_shot_root_dir,
        massive_neutrinos=True,
        idx_range=range(
            args.n_shots + args.n_shots_val,
            args.n_shots + args.n_shots_val + args.n_shots_test,
        ),
        cosmological_parameters=args.cosmological_parameters
        + args.few_shot_cosmological_parameters,
    )
    few_shot_model = cTransfer(
        summarizer=summarizer,
        n_features=len(
            args.cosmological_parameters + args.few_shot_cosmological_parameters
        ),
        n_transforms=args.n_transforms,
        freeze_summarizer=True if args.n_baseline > 0 else False,
        phase='few_shot'
    )
    if args.n_shots > 0:
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(args.output_dir) / f"{run_name}_few_shot",
            save_top_k=1,
            monitor="few_shot_val_loss",
            save_last=True,
            auto_insert_metric_name=True,
            every_n_train_steps=args.save_every,
        )
        early_callback = EarlyStopping(monitor="few_shot_val_loss", mode="min")
        few_shot_trainer = setup_trainer(
            args,
            args.few_shot_total_steps,
            wandb_logger,
            checkpoint_callback,
            early_callback,
        )
        few_shot_trainer.fit(
            model=few_shot_model,
            train_dataloaders=few_shot_train_loader,
            val_dataloaders=few_shot_val_loader,
        )
        few_shot_trainer.test(dataloaders=few_shot_test_loader)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    train(args)
