from typing import Optional, Callable, Tuple, List
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torchvision.datasets import VisionDataset
from torchio.transforms import Resize

QUIJOTE_STD = {
    "256": 1.96,
}


class Quijote(VisionDataset):
    def __init__(
        self,
        root: str = "/n/holyscratch01/iaifi_lab/Lab/quijote_large/density_fields/",
        redshift: float = 0.0,
        transform: Optional[Callable] = None,
        cosmological_parameters=["Omega_m", "Omega_b", "h", "n_s", "sigma8"],
        resolution: int = 256,
        original_resolution: int = 256,
        massive_neutrinos: bool = False,
        idx_list: Optional[List[int]] = None,
    ):
        super().__init__(
            root,
            transform=transform,
        )
        self.root = Path(self.root)
        self.redshift = redshift
        self.idx_list = idx_list
        self.massive_neutrinos = massive_neutrinos
        if resolution != original_resolution:
            self.resize = Resize((resolution, resolution, resolution))
        else:
            self.resize = None
        self.resolution = resolution
        self._load_cosmologies(cosmological_parameters=cosmological_parameters)

    def __len__(
        self,
    ):
        return len(self.idx_list)

    def _load_cosmologies(
        self, cosmological_parameters=["Omega_m", "Omega_b", "h", "n_s", "sigma_8"]
    ):
        if self.massive_neutrinos:
            cosmo_url = "https://raw.githubusercontent.com/franciscovillaescusa/Quijote-simulations/master/latin_hypercube_nwLH/latin_hypercube_params.txt"
            column_names = ["Omega_m", "Omega_b", "h", "n_s", "sigma_8", "M_nu", "w"]
        else:
            cosmo_url = "https://raw.githubusercontent.com/franciscovillaescusa/Quijote-simulations/master/BSQ/BSQ_params.txt"
            column_names = ["Omega_m", "Omega_b", "h", "n_s", "sigma_8"]
        self.cosmological_parameters = cosmological_parameters
        self.cosmologies = pd.read_csv(
            cosmo_url,
            sep=" ",
            names=column_names,
            skiprows=1,
            header=None,
        )[self.cosmological_parameters].values
        self.cosmologies = torch.from_numpy(self.cosmologies.copy()).float()

    def read_density(
        self,
        idx,
    ):
        if self.massive_neutrinos:
            density = np.load(
                self.root / f"df_idx{idx}_m_CIC_z={self.redshift:.2f}.npy",
            )
        else:
            density = np.load(
                self.root / f"{idx}/df_m_CIC_z={self.redshift:.2f}.npy",
            )
        if self.resize is not None:
            density = self.resize(density[None]).squeeze()
        if not self.massive_neutrinos:
            density = density / density.mean() - 1
        # divide by the variance at that redshift
        density /= QUIJOTE_STD[f"{self.resolution}"]
        return torch.from_numpy(density).float()

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self.idx_list[index]
        delta = self.read_density(
            idx,
        ).unsqueeze(0)
        if self.transform is not None:
            delta = self.transform(delta)
        cosmology = self.cosmologies[idx]
        return delta, cosmology
