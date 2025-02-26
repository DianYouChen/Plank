import os, sys, glob
import numpy as np
from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt

from const import (h, k, c, mu2m, tera)

def compute_radiance(mode):
    """Decorator to compute spectral radiance based on the selected mode."""
    def operator(func):
        def wrapper(self, spectral_or_frequency: np.array):
            values = np.array(spectral_or_frequency) * (mu2m if mode in ['wavelength', 'wien', 'rayleigh'] else tera)
            temperatures = np.array(self._Temp)[:, None]  # Convert temperature list to a 2D array

            if mode == 'wavelength':  # Planck's Law (λ)
                B = (2 * h * c**2 / values[None, :]**5) * \
                    np.reciprocal(np.exp((h * c) / (values[None, :] * k * temperatures)) - 1) * mu2m
            elif mode == 'frequency':  # Planck's Law (ν)
                B = (2 * h * values[None, :]**3 / c**2) * \
                    np.reciprocal(np.exp((h * values[None, :]) / (k * temperatures)) - 1)
            elif mode == 'wien':  # Wien's Approximation
                B = (2 * h * c**2 / values[None, :]**5) * \
                    np.reciprocal(np.exp((h * c) / (values[None, :] * k * temperatures))) * mu2m
            elif mode == 'rayleigh':  # Rayleigh-Jeans Law
                B = (2 * c * k * temperatures / values[None, :]**4) * mu2m
            else:
                raise ValueError("Invalid mode specified.")

            return B  # Shape: (num_temps, num_spectral_points)
        
        return wrapper
    return operator

class PlankSolver:
    def __init__(self, temperature: list):
        self._Temp = temperature

    @compute_radiance('wavelength')
    def get_plank_lamda(self, spectral: np.array) -> np.ndarray:
        """Calculate spectral radiance using Planck's law (wavelength-based)."""
        pass  # Handled by decorator

    @compute_radiance('frequency')
    def get_plank_nu(self, frequency: np.array) -> np.ndarray:
        """Calculate spectral radiance using Planck's law (frequency-based)."""
        pass  # Handled by decorator

    @compute_radiance('wien')
    def get_Wien(self, spectral: np.array) -> np.ndarray:
        """Approximate spectral radiance using Wien's approximation."""
        pass  # Handled by decorator

    @compute_radiance('rayleigh')
    def get_RayJean(self, spectral: np.array) -> np.ndarray:
        """Approximate spectral radiance using the Rayleigh-Jeans law."""
        pass  # Handled by decorator


def drawer_lamda(distribution: list,
                 spectral: np.array,
                 temperature: list
                 )-> None:

    assert len(distribution) == len(temperature), "Size inconsistency."

    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5), dpi=200, facecolor="w")
    for curve in distribution:
        ax.plot(spectral, curve.squeeze(), lw=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=0.1, right=spectral[-1])
    ax.set_ylim(bottom=10**0, top=10**8)
    ax.set_xticks(list(map(lambda n: 10**n, [ _ for _ in range(-1,2)])))
    ax.set_xticklabels(list(map(lambda m: 10**m, [ _ for _ in range(-1,2)])),fontsize=12.8)
    ax.set_yticks(list(map(lambda n: 10**n, [ _ for _ in range(0,9)])))
    ax.set_yticklabels(list(map(lambda n: f"10$^{n}$", [ _ for _ in range(0,9)])),fontsize=12.8)
    ax.set_xlabel("Wavelength $\lambda$ ($\mu$m)", fontsize=13.8)
    ax.set_ylabel("B$_\lambda$(T) (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)", fontsize=13.8)
    ax.legend(list(map(lambda t: str(t)+"K", [ _ for _ in temperature])), 
              loc="best", fontsize=15, handlelength=2.5)


def drawer_lamda_nu(spectral:  np.array,
                    frequency: np.array,
                    temperature: list,
                    varns=['lamda','nu'],
                    **distribution,
                    )-> None:

    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5), dpi=200, facecolor="w")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    for varn in varns:
        if varn == "lamda":
            for arc in distribution[varn]:
                ax.plot(spectral, arc.squeeze(), lw=3, color="C1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(left=0.3, right=spectral[-1])
        ax.set_ylim(bottom=1.5*10**7, top=3*10**7)
        ax.set_xticks(list(map(lambda n: n/10, [ _ for _ in range(3,11)])))
        ax.set_xticklabels(list(map(lambda m: m/10, [ _ for _ in range(3,11)])),fontsize=12.8)
        ax.set_yticks(list(map(lambda n: n*10**7, [1.5, 2, 3])))
        ax.set_yticklabels(list(map(lambda m: f"{m}x10$^7$", [1.5, 2, 3])),fontsize=12.8)
        ax.set_xlabel("Wavelength $\lambda$ ($\mu$m)", fontsize=13.8, color="C1")
        ax.set_ylabel("B$_\lambda$(T) (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)", fontsize=13.8, color="C1")
        ax.tick_params(axis='x', colors="C1")
        ax.tick_params(axis='y', colors="C1")
        
        if varn == "nu":
            for curve in distribution[varn]:
                ax2.plot(frequency, curve.squeeze(), lw=3, color="C2")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.xaxis.tick_top()
        ax2.yaxis.tick_right()
        ax2.set_xlim(left=10**11, right=10**16)
        ax2.set_ylim(bottom=10**-10, top=10**-7)
        ax2.xaxis.set_label_position('top') 
        ax2.yaxis.set_label_position('right')
        ax2.set_xticks(list(map(lambda n: 10**n, [ _ for _ in range(11,17)])))
        ax2.set_xticklabels(list(map(lambda m: f"10$^{m//10:0d}$"+f"$^{m%10}$" if m >= 10 else f"10$^{m%10}$"
                                    ,[ _ for _ in range(11,17)])), fontsize=12.8)
        ax2.set_yticks(list(map(lambda n: 10**n, [ _ for _ in range(-10,-6)])))
        ax2.set_yticklabels(["10$^{-10}$","10$^{-9}$","10$^{-8}$","10$^{-7}$"],fontsize=12.8)
        ax2.set_xlabel("Frequency (Hz)", fontsize=13.8 ,color="C2")
        ax2.set_ylabel("B$_\\nu$(T) (J s$^{-1}$ m$^{-2}$ Hz$^{-1}$ ster$^{-1}$)", fontsize=13.8, color="C2")
        ax2.tick_params(axis='x', colors="C2")
        ax2.tick_params(axis='y', colors="C2")

class Preprocess:

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    @staticmethod
    def get_max_value(x, y):
        return max(x, y)

def drawer_approximations(distribution: list,
                          spectral: np.array,
                          *methods,
                          colors=None,
                          )-> None:

    N = len(distribution)
    if colors == None:
        colors = ["C" + str(i) for i in range(N)]
    
    # Calculate the difference
    locs = []
    for i, curve in enumerate(distribution, start=1):
        if i < len(distribution):
            diff = np.absolute(distribution[i]-distribution[0])/distribution[0]
            idx  = Preprocess.find_nearest(diff.squeeze(), 0.05)
            locs.append(idx)
    
    # Plot figures for variant approximations
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5), dpi=200, facecolor="w")
    for k, curve in enumerate(distribution):
        ax.plot(spectral, 
                curve.squeeze(), 
                lw=3, 
                color=colors[k], 
                zorder=len(distribution)-k)
        
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=0.1, right=spectral[-1])
    ax.set_ylim(bottom=10**3, top=10**8)
    ax.set_xticks(list(map(lambda n: 10**n, [ _ for _ in range(-1,2)])))
    ax.set_xticklabels(list(map(lambda m: 10**m, [ _ for _ in range(-1,2)])),fontsize=12.8)
    ax.set_yticks(list(map(lambda n: 10**n, [ _ for _ in range(2,9)])))
    ax.set_yticklabels(list(map(lambda n: f"10$^{n}$", [ _ for _ in range(2,9)])),fontsize=12.8)
    ax.set_xlabel("Wavelength $\lambda$ ($\mu$m)", fontsize=13.8)
    ax.set_ylabel("B$_\lambda$(T) (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)", fontsize=13.8)
    ax.legend(labels = methods, loc="best", fontsize=15, handlelength=2.5)

    markers = ['X', 'D',]
    for ii, _idx in enumerate(locs, start=1):
        _dist = distribution[ii].squeeze()
        ax.plot(spectral[_idx], _dist[_idx], 
                markers[(ii-1) % 10], 
                color=colors[ii],
                markersize=10,
                zorder=5)
    
    ax.text(spectral[locs[0]-5],
            distribution[0].squeeze()[locs[0]+10],
            f"{spectral[locs[0]]:.2f}"+"$\mu$m",
            horizontalalignment="center",
            verticalalignment="center",
            color=colors[1],
            fontsize=18,
            fontweight="normal",
            )
    
    ax.text(spectral[locs[1]-200],
            distribution[1].squeeze()[locs[1]-270],
            f"{spectral[locs[1]]:.2f}"+"$\mu$m",
            horizontalalignment="center",
            verticalalignment="center",
            color=colors[2],
            fontsize=18,
            fontweight="normal",
            zorder=7
            )
    
 


"""

###########################################################
########### Original version of PlankSolver ###############
###########################################################

class plankSolver():

    def __init__(self, 
                 temperature: list, 
                ):
        self._Temp = temperature

    def get_plank_lamda(self, spectral: np.array) -> list:
        B_lamda = []
        wave_lengh = list(spectral*mu2m) # from μm to m
        for i, temp in enumerate(self._Temp):
            B_lamda.append(list(map(lambda lam: 
                                    ((2*h*c**2)/(lam**5))*np.reciprocal(np.exp((h*c)/(lam*k*temp))-1)*mu2m, 
                                    wave_lengh)))
        return B_lamda
    
    def get_plank_nu(self, frequency: np.array) -> list:
        B_nu = []
        _nu = list(frequency*tera) # from THz to Hz
        for i, temp in enumerate(self._Temp):
            B_nu.append(list(map(lambda nu: 
                                    ((2*h*nu**3)/(c**2))*np.reciprocal(np.exp((h*nu)/(k*temp))-1), 
                                    _nu)))
        return B_nu
    
    def get_Wien(self, spectral: np.array) -> list:
        Wien_lamda = []
        wave_lengh = list(spectral*mu2m) # from μm to m
        for i, temp in enumerate(self._Temp):
            Wien_lamda.append(list(map(lambda lam: 
                                    ((2*h*c**2)/(lam**5))*np.reciprocal(np.exp((h*c)/(lam*k*temp)))*mu2m, 
                                    wave_lengh)))
    
    def get_RayJean(self, spectral: np.array) -> list:
        Wien_lamda = []
        wave_lengh = list(spectral*mu2m) # from μm to m
        for i, temp in enumerate(self._Temp):
            Wien_lamda.append(list(map(lambda lam: 
                                    ((2*c*k)/(lam**4))*temp))*mu2m, 
                                    wave_lengh)
"""

