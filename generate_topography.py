"""
EEG Topography Generator
========================

Generates configurable EEG visualizations from per-electrode absolute power data:
  - Topographic maps (individual and combined)
  - Regional bar charts
  - Electrode × Band heatmaps

All options are driven by config.json. See README for schema details.
"""

import logging
import math
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mne
from mne.viz import plot_topomap

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


DEFAULTS = {
    "excel_file": "",
    "output_dir_python": "topography_plots",
    "electrodes": ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz',
                   'C4','T4','T5','P3','Pz','P4','T6','O1','O2'],
    "bands": ['delta','theta','alpha','beta'],
    "regions": {
        "Frontal":  ["Fp1","Fp2","F7","F3","Fz","F4","F8"],
        "Central":  ["C3","Cz","C4"],
        "Temporal": ["T3","T4","T5","T6"],
        "Parietal": ["P3","Pz","P4"],
        "Occipital":["O1","O2"]
    },
    "visualization": {
        "colormap": "viridis",
        "dpi": 300,
        "output_format": "png",
        "num_contours": 6,
        "font_size_title": 16,
        "font_size_labels": 10,
        "figure_size": [8, 8],
        "electrode_style": "labels",
        "color_scale": "auto",
        "power_label": "Absolute Power (µV²)",
        "plots": {
            "topoplot": True,
            "topoplot_combined": True,
            "regional_bar": True,
            "heatmap": True
        }
    }
}

def load_config(config_path: str = "config.json") -> dict:
    """Loads config with fallbacks to defaults for every field."""
    cfg = DEFAULTS.copy()
    path = Path(config_path)
    if path.exists():
        with open(path, 'r') as f:
            user_cfg = json.load(f)
        for key in ['excel_file','output_dir_python','electrodes','bands','eeglab_path','regions']:
            if key in user_cfg:
                cfg[key] = user_cfg[key]

        if 'visualization' in user_cfg:
            for key, val in user_cfg['visualization'].items():
                if key == 'plots' and isinstance(val, dict):
                    cfg['visualization']['plots'].update(val)
                else:
                    cfg['visualization'][key] = val
    else:
        logger.warning(f"Config file {config_path} not found. Using defaults.")
    return cfg


def detect_data_file(cfg_path: str) -> str:
    """Auto-detects a data file (.xlsx, .csv, .txt) in the working directory."""
    if cfg_path:
        return cfg_path
    supported = ['.xlsx', '.csv', '.txt']
    candidates = [f for f in Path('.').iterdir()
                  if f.is_file() and f.suffix.lower() in supported]
    if len(candidates) == 1:
        logger.info(f"Auto-detected data file: {candidates[0].name}")
        return str(candidates[0])
    if len(candidates) == 0:
        raise FileNotFoundError("No data file found. Place an .xlsx, .csv, or .txt file "
                                "in the working directory, or set 'excel_file' in config.json.")
    names = ', '.join(f.name for f in candidates)
    raise FileNotFoundError(f"Multiple data files found: {names}. "
                            "Set 'excel_file' in config.json to specify which one to use.")

def load_eeg_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    logger.info(f"Loading data from {path.name}...")
    ext = path.suffix.lower()
    if ext == '.xlsx':
        return pd.read_excel(file_path)
    elif ext == '.csv':
        return pd.read_csv(file_path)
    elif ext == '.txt':
        return pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def setup_mne_info(electrodes: List[str]) -> mne.Info:
    """Creates an MNE Info object with a standard 10-20 montage."""
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=electrodes, sfreq=256, ch_types='eeg')
    info.set_montage(montage)
    return info

def compute_grand_averages(df: pd.DataFrame, bands: List[str],
                           electrodes: List[str]) -> Dict[str, np.ndarray]:
    band_averages = {}
    logger.info("Computing grand average power per electrode...")
    for band in bands:
        cols = [f"{band}_{ch}" for ch in electrodes]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            logger.warning(f"Missing columns for {band}: {missing}")
            continue
        means = df[cols].mean().values
        band_averages[band] = means
        logger.info(f"  {band.capitalize()}: min={means.min():.3f}, max={means.max():.3f}")
    return band_averages


def _vis(cfg, key):
    return cfg['visualization'].get(key, DEFAULTS['visualization'][key])

def _plots_enabled(cfg, plot_name):
    return cfg['visualization'].get('plots', {}).get(plot_name, False)

def _save_fig(fig, output_path: Path, name: str, cfg: dict):
    fmt = _vis(cfg, 'output_format')
    filepath = output_path / f"{name}.{fmt}"
    fig.savefig(filepath, dpi=_vis(cfg, 'dpi'), bbox_inches='tight')
    logger.info(f"  Saved: {filepath.name}")
    plt.close(fig)


def plot_single_topomap(data, info, title, ax, cfg):
    cmap = _vis(cfg, 'colormap')
    contours = _vis(cfg, 'num_contours')
    style = _vis(cfg, 'electrode_style')
    title_size = _vis(cfg, 'font_size_title')

    sensor_flag = style != 'off'
    names = info.ch_names if style == 'labels' else None

    im, _ = plot_topomap(
        data, info, axes=ax, show=False,
        contours=contours, cmap=cmap,
        vlim=(data.min(), data.max()),
        sensors=sensor_flag, names=names
    )
    ax.set_title(title, fontsize=title_size, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(_vis(cfg, 'power_label'))

def generate_topoplots(averages, info, output_path, cfg):
    bands = list(averages.keys())
    fig_size = _vis(cfg, 'figure_size')


    if _plots_enabled(cfg, 'topoplot'):
        for band, data in averages.items():
            fig, ax = plt.subplots(figsize=fig_size)
            plot_single_topomap(data, info, f"{band.capitalize()} Band", ax, cfg)
            _save_fig(fig, output_path, f"{band}_topomap", cfg)


    if _plots_enabled(cfg, 'topoplot_combined'):
        n = len(bands)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(fig_size[0]*ncols*0.9, fig_size[1]*nrows*0.9))
        fig.suptitle('Grand Average EEG Topography', fontsize=20, fontweight='bold')
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for idx, band in enumerate(bands):
            plot_single_topomap(averages[band], info,
                                f"{band.capitalize()} Band",
                                axes_flat[idx], cfg)
        for idx in range(len(bands), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        _save_fig(fig, output_path, "combined_topography", cfg)


def generate_regional_bar_chart(averages, electrodes, regions, output_path, cfg):
    bands = list(averages.keys())
    region_names = list(regions.keys())
    title_size = _vis(cfg, 'font_size_title')
    label_size = _vis(cfg, 'font_size_labels')
    fig_size = _vis(cfg, 'figure_size')


    region_means = {band: [] for band in bands}
    for band in bands:
        data = averages[band]
        for region_electrodes in regions.values():
            indices = [electrodes.index(e) for e in region_electrodes if e in electrodes]
            region_means[band].append(np.mean(data[indices]) if indices else 0)


    x = np.arange(len(region_names))
    n_bands = len(bands)
    width = 0.8 / n_bands
    cmap = matplotlib.colormaps.get_cmap(_vis(cfg, 'colormap'))
    colors = [cmap(i / max(n_bands - 1, 1)) for i in range(n_bands)]

    fig, ax = plt.subplots(figsize=(fig_size[0] * 1.4, fig_size[1]))
    for i, band in enumerate(bands):
        offset = (i - n_bands / 2 + 0.5) * width
        bars = ax.bar(x + offset, region_means[band], width, label=band.capitalize(),
                      color=colors[i], edgecolor='white', linewidth=0.5)

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}', ha='center', va='bottom',
                    fontsize=label_size - 2)

    ax.set_xlabel('Brain Region', fontsize=label_size)
    ax.set_ylabel(_vis(cfg, 'power_label').replace('Absolute ', 'Mean '), fontsize=label_size)
    ax.set_title('Regional Power Distribution', fontsize=title_size, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(region_names, fontsize=label_size)
    ax.legend(fontsize=label_size)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, output_path, "regional_bar_chart", cfg)


def generate_heatmap(averages, electrodes, output_path, cfg):
    bands = list(averages.keys())
    title_size = _vis(cfg, 'font_size_title')
    label_size = _vis(cfg, 'font_size_labels')
    fig_size = _vis(cfg, 'figure_size')


    matrix = np.column_stack([averages[b] for b in bands])

    fig, ax = plt.subplots(figsize=(fig_size[0] * 0.8, fig_size[1] * 1.2))
    im = ax.imshow(matrix, cmap=_vis(cfg, 'colormap'), aspect='auto')


    for i in range(len(electrodes)):
        for j in range(len(bands)):
            val = matrix[i, j]

            norm_val = (val - matrix.min()) / (matrix.max() - matrix.min())
            text_color = 'white' if norm_val > 0.6 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=label_size - 2, color=text_color)

    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.capitalize() for b in bands], fontsize=label_size)
    ax.set_yticks(range(len(electrodes)))
    ax.set_yticklabels(electrodes, fontsize=label_size)
    ax.set_title('Electrode × Band Power Matrix', fontsize=title_size, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(_vis(cfg, 'power_label'))
    plt.tight_layout()
    _save_fig(fig, output_path, "electrode_band_heatmap", cfg)


def run_topography_pipeline():
    cfg = load_config()
    logger.info(f"Starting EEG Topography Pipeline (MNE {mne.__version__})")

    try:
        data_file = detect_data_file(cfg['excel_file'])
        df = load_eeg_data(data_file)
        electrodes = cfg['electrodes']
        info = setup_mne_info(electrodes)
        averages = compute_grand_averages(df, cfg['bands'], electrodes)

        output_path = Path(cfg['output_dir_python'])
        output_path.mkdir(exist_ok=True)

        fmt = _vis(cfg, 'output_format')
        cmap_name = _vis(cfg, 'colormap')
        try:
            matplotlib.colormaps.get_cmap(cmap_name)
        except (KeyError, ValueError):
            logger.warning(f"Unknown colormap '{cmap_name}', falling back to 'viridis'.")
            cfg['visualization']['colormap'] = 'viridis'
        logger.info(f"Generating plots in '{output_path.absolute()}' (format: {fmt})...")


        generate_topoplots(averages, info, output_path, cfg)


        if _plots_enabled(cfg, 'regional_bar'):
            generate_regional_bar_chart(averages, electrodes,
                                        cfg.get('regions', DEFAULTS['regions']),
                                        output_path, cfg)


        if _plots_enabled(cfg, 'heatmap'):
            generate_heatmap(averages, electrodes, output_path, cfg)

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_topography_pipeline()
    plt.close('all')
