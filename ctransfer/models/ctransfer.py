import torch
import zuko
import matplotlib.pyplot as plt
import numpy as np
import wandb
import io
from PIL import Image
from lightning import LightningModule


def log_matplotlib_figure(figure_label: str):
    """log a matplotlib figure to wandb, avoiding plotly

    Args:
        figure_label (str): label for figure
    """
    # Save plot to a buffer, otherwise wandb does ugly plotly
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        dpi=300,
    )
    buf.seek(0)
    image = Image.open(buf)
    # Log the plot to wandb
    wandb.log({f"{figure_label}": wandb.Image(image)})

class cTransfer(LightningModule):
    """
    A PyTorch Lightning Module for conditional transfer learning using 
    a summarizer and a Masked Autoregressive Flow (MAF) density estimator.
    
    Attributes:
        summarizer (nn.Module): The summarizer model used to extract features.
        n_features (int): The number of features in the input data.
        freeze_summarizer (bool): A flag indicating whether to freeze the summarizer during training.
        n_transforms (int): The number of transforms for the MAF density estimator.
        learning_rate (float): The learning rate for the optimizer.
        scheduler_patience (int): The patience parameter for the learning rate scheduler.
        validation_step_outputs (list): A list to store validation step outputs.
        density_estimator (zuko.flows.MAF): The MAF density estimator.
    """

    def __init__(
        self,
        summarizer: torch.nn.Module,
        summary_dim: int,
        n_features: int,
        freeze_summarizer: bool = False,
        n_transforms: int = 5,
        learning_rate: float = 1.0e-4,
        scheduler_patience: int = 10,
        phase: str = 'baseline',
    ):
        """
        Initializes the cTransfer module.

        Args:
            summarizer (torch.nn.Module): The summarizer model used to extract features.
            n_features (int): The number of features in the input data.
            freeze_summarizer (bool): A flag indicating whether to freeze the summarizer during training.
            n_transforms (int): The number of transforms for the MAF density estimator.
            learning_rate (float): The learning rate for the optimizer.
            scheduler_patience (int): The patience parameter for the learning rate scheduler.
            phase (str): The phase of training, either "baseline" or "few_shot".

        """
        super().__init__()

        self.save_hyperparameters(ignore=["summarizer"])
        self.summary_dim = summary_dim
        #self.save_hyperparameters({"summarizer_hparams": summarizer.hparams})
        self.freeze_summarizer = freeze_summarizer
        self.validation_step_outputs = []
        self.summarizer = summarizer
        self.density_estimator = zuko.flows.MAF(
            features=n_features,
            context=self.summary_dim,
            transforms=n_transforms,
        )
        self.phase = phase

    def setup(self, stage: str):
        """
        Sets up the module for training or evaluation.

        Args:
            stage (str): The stage of the setup ('fit', 'validate', 'test', or 'predict').
        """
        if stage == "fit":
            self.summarizer = self.summarizer.to(self.device)
            self.density_estimator = self.density_estimator.to(self.device)
            if self.freeze_summarizer:
                for param in self.summarizer.parameters():
                    param.requires_grad = False

    def get_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for the given inputs and targets.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed loss.
        """
        return -self.density_estimator(x).log_prob(y).mean()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch (torch.Tensor): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        x, y = batch
        embedding = self.summarizer(x)
        loss = self.get_loss(embedding, y)
        self.log(f"{self.phase}_train_loss", loss, prog_bar=True, sync_dist=True, logger=True,)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Performs a single validation step.

        Args:
            batch (torch.Tensor): The input batch.
            batch_idx (int): The index of the batch.
        """
        x, y = batch
        embedding = self.summarizer(x)
        loss = self.get_loss(x=embedding, y=y)
        output_dict = {
            f"{self.phase}_val_loss": loss,
            "batch": batch,
        }
        self.log(f"{self.phase}_val_loss", loss, prog_bar=True, sync_dist=True, logger=True,)

        self.validation_step_outputs.append(output_dict)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Performs a single test step.

        Args:
            batch (torch.Tensor): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        x, y = batch
        embedding = self.summarizer(x)
        loss = self.get_loss(embedding, y)
        self.log(f"{self.phase}_test_loss", loss, prog_bar=True, sync_dist=True, logger=True,)
        return loss

    def _log_pred_vs_true(self, batch, n_samples=(100,)):
        x, y = batch
        embedding = self.summarizer(x)
        predictions = self.density_estimator(embedding).sample(n_samples)
        fig, ax = plt.subplots(ncols=y.shape[-1], figsize=(5*y.shape[-1], 5))
        for i in range(y.shape[-1]):
            ax[i].plot(
                y[:,i].cpu().numpy(), 
                y[:,i].cpu().numpy(), 
                linestyle='dashed',
                color='lightgray'
            )
            ax[i].errorbar(
                y[:,i].cpu().numpy(), 
                np.mean(predictions[...,i].cpu().numpy(),axis=0), 
                yerr=np.std(predictions[...,i].cpu().numpy(),axis=0), 
                alpha=0.5,
                linestyle='None',
                markersize=3,
                fmt='o',
            
            )
        log_matplotlib_figure(f"true vs pred | {self.phase}")
        plt.close()


    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to aggregate results.
        """
        mean_val_loss = torch.stack(
            [output_dict[f"{self.phase}_val_loss"] for output_dict in self.validation_step_outputs]
        ).mean()
        self.log(
            f"mean_{self.phase}_val_loss",
            mean_val_loss,
            prog_bar=True,
            sync_dist=True,
        )
        # Add if want to add any validation figures
        batch =  self.validation_step_outputs[0]['batch']
        self._log_pred_vs_true(batch)
        self.validation_step_outputs.clear()

    def configure_optimizers(self) -> dict:
        """
        Configures the optimizers and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and scheduler.
        """
        if self.freeze_summarizer:
            parameters = list(self.density_estimator.parameters())
        else:
            parameters = list(self.summarizer.parameters()) + list(
                self.density_estimator.parameters()
            )

        optimizer = torch.optim.AdamW(parameters, lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.hparams.scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": f"{self.phase}_val_loss",
        }
