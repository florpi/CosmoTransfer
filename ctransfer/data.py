from typing import Optional, Callable, Tuple, List
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torchvision.datasets import VisionDataset
from torchio.transforms import Resize

QUIJOTE_STATS_3D = {
    "256": [8., 18.],
}
QUIJOTE_STATS_2D = {
    "256": [12.6, 0.275],
}
COSMO_STATS = {
    'Omega_m': [0.3, 0.1155],
    'Omega_b': [0.05, 0.0115],
    'h': [0.7, 0.1155],
    'n_s': [1., 0.1155],
    'sigma_8': [0.8, 0.1155],
    'M_nu': [0.5050,0.2859],
    'w': [-1.,0.1732],
}


class Quijote(VisionDataset):
    def __init__(
        self,
        #root: str = "/n/holyscratch01/iaifi_lab/Lab/quijote_large/density_fields/",
        root: str = '/n/holystore01/LABS/iaifi_lab/Users/ccuestalazaro/cosmo_transfer/BSQ',
        redshift: float = 0.0,
        transform: Optional[Callable] = None,
        cosmological_parameters=["Omega_m", "Omega_b", "h", "n_s", "sigma8"],
        resolution: int = 256,
        original_resolution: int = 256,
        massive_neutrinos: bool = False,
        fnl: bool = False,
        idx_list: Optional[List[int]] = None,
        dim: str = '2d',
    ):
        super().__init__(
            root,
            transform=transform,
        )
        self.root = Path(self.root)
        self.redshift = redshift
        self.idx_list = idx_list
        self.massive_neutrinos = massive_neutrinos
        self.fnl = fnl
        if resolution != original_resolution:
            self.resize = Resize((resolution, resolution, resolution))
        else:
            self.resize = None
        self.resolution = resolution
        self.dim = dim
        self.norm_dict = QUIJOTE_STATS_2D if self.dim == '2d' else QUIJOTE_STATS_3D
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
        elif self.fnl:
            cosmo_url = "https://raw.githubusercontent.com/franciscovillaescusa/Quijote-simulations/master//latin_hypercube_params.txt"
            column_names = ['f_NL_equilateral', 'Omega_m', 'h', 'n_s', 'sigma_8']
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
        # standarize cosmology params
        self.cosmologies = (self.cosmologies - np.array([COSMO_STATS[cosmo][0] for cosmo in self.cosmological_parameters])) / np.array([COSMO_STATS[cosmo][1] for cosmo in self.cosmological_parameters])
        self.cosmologies = torch.from_numpy(self.cosmologies.copy()).float()

    def read_density(
        self,
        idx,
    ):
        if self.dim == '3d':
            density = np.load(
                self.root / f"{idx}/df_m_CIC_z={self.redshift:.2f}.npy",
            )
        elif self.dim == '2d':
            # read in a random view from the 
            view = np.random.randint(0, 24)
            try:
                density = np.log10(np.load(
                    self.root / f"{idx}/mass_m_CIC_z={self.redshift:.2f}.npy",
                )[view])
            except:
                print(f'ERROR READING file {self.root / f"{idx}/mass_m_CIC_z={self.redshift:.2f}.npy"}')
                density = np.log10(np.load(
                    self.root / f"0/mass_m_CIC_z={self.redshift:.2f}.npy",
                )[view])

        if self.resize is not None:
            density = self.resize(density[None]).squeeze()
        #density = density / density.mean() - 1
        density = (density - self.norm_dict[f"{self.resolution}"][0]) / self.norm_dict[f"{self.resolution}"][1]
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
