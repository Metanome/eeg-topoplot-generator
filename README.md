# EEG Topography Visualization

Generates scalp topography plots, regional power bar charts, and electrode×band heatmaps from qEEG power data. Supports absolute power, relative power, spectral density, or any other band-level metric. Dual implementations in **Python** (MNE) and **MATLAB** (EEGLAB) produce equivalent outputs from a shared configuration file.

## Requirements

### Python
- Python 3.10+
- Dependencies: `pip install mne numpy pandas matplotlib openpyxl`

### MATLAB
- MATLAB R2020b+
- [EEGLAB](https://sccn.ucsd.edu/eeglab/) toolbox installed and on MATLAB path

## Quick Start

1. Place your data file (`.xlsx`, `.csv`, or `.txt`) in the `eeg-topoplot-generator/` directory
2. Run either script:

### Python
```bash
python generate_topography.py
```

### MATLAB
```matlab
generate_topography_eeglab
```

Both scripts read `config.json` from the working directory. If the file is missing, built-in defaults are used. If `excel_file` is empty or omitted, the script auto-detects a single data file in the working directory.

## Input Data

A tabular file (`.xlsx`, `.csv`, or tab-delimited `.txt`) with columns named `{band}_{electrode}` (e.g., `alpha_Fp1`, `delta_O2`). Each row is a participant; values are grand-averaged across all rows.

## Output

| Plot Type              | Python Filename                | MATLAB Filename                       |
| ---------------------- | ------------------------------ | ------------------------------------- |
| Individual topoplot    | `{band}_topomap.{fmt}`         | `{band}_eeglab_topomap.{fmt}`         |
| Combined topoplot      | `combined_topography.{fmt}`    | `combined_topography_eeglab.{fmt}`    |
| Regional bar chart     | `regional_bar_chart.{fmt}`     | `regional_bar_chart_eeglab.{fmt}`     |
| Electrode×band heatmap | `electrode_band_heatmap.{fmt}` | `electrode_band_heatmap_eeglab.{fmt}` |

## Configuration Reference

All parameters are set in `config.json`. Every field is optional — defaults are used for anything omitted.

### Data & Paths

| Key                 | Type     | Default                            | Description                                                                                                 |
| ------------------- | -------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `excel_file`        | string   | `""` (auto-detect)                 | Input data file path. If empty, auto-detects a single `.xlsx`/`.csv`/`.txt` in the working directory.       |
| `output_dir_python` | string   | `"topography_plots"`               | Python output directory                                                                                     |
| `output_dir_matlab` | string   | `"topography_plots_eeglab"`        | MATLAB output directory                                                                                     |
| `eeglab_path`       | string   | `""`                               | EEGLAB install path (auto-detected if empty)                                                                |
| `electrodes`        | string[] | 19 standard 10-20 labels           | Electrode labels matching data column names                                                                 |
| `bands`             | string[] | `["delta","theta","alpha","beta"]` | Frequency band names matching data column prefixes. Any number of bands is supported (e.g., add `"gamma"`). |

### Regions

The `regions` object maps region names to electrode lists, used by the bar chart:

```json
"regions": {
    "Frontal":  ["Fp1","Fp2","F7","F3","Fz","F4","F8"],
    "Central":  ["C3","Cz","C4"],
    "Temporal": ["T3","T4","T5","T6"],
    "Parietal": ["P3","Pz","P4"],
    "Occipital":["O1","O2"]
}
```

### Visualization

All fields live under the `"visualization"` object:

| Key                | Type   | Default                  | Description                                                                                                         |
| ------------------ | ------ | ------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `colormap`         | string | `"viridis"`              | Matplotlib/MATLAB colormap name. Invalid names fall back to `viridis` (Python) or `parula` (MATLAB) with a warning. |
| `dpi`              | int    | `300`                    | Output resolution (PNG only)                                                                                        |
| `output_format`    | string | `"png"`                  | File format: `png`, `svg`, or `pdf`                                                                                 |
| `num_contours`     | int    | `6`                      | Number of contour lines on topoplots                                                                                |
| `font_size_title`  | int    | `16`                     | Title font size (pt)                                                                                                |
| `font_size_labels` | int    | `10`                     | Axis label / annotation font size (pt)                                                                              |
| `figure_size`      | [w, h] | `[8, 8]`                 | Figure dimensions in inches (Python only)                                                                           |
| `electrode_style`  | string | `"labels"`               | Topoplot electrode display: `"labels"`, `"dots"`/`"on"`, or `"off"`                                                 |
| `color_scale`      | string | `"auto"`                 | Color axis scaling (reserved for future use)                                                                        |
| `power_label`      | string | `"Absolute Power (µV²)"` | Axis label for colorbars and y-axes. Change for relative power, spectral density, etc.                              |

### Plot Toggles

Under `"visualization" → "plots"`, each boolean enables/disables a plot type:

| Key                 | Default | Description                       |
| ------------------- | ------- | --------------------------------- |
| `topoplot`          | `true`  | Individual topoplot per band      |
| `topoplot_combined` | `true`  | Combined topoplot grid (1 file)   |
| `regional_bar`      | `true`  | Regional power bar chart (1 file) |
| `heatmap`           | `true`  | Electrode×band heatmap (1 file)   |

## Platform Differences

| Behavior                  | Python                     | MATLAB                                 |
| ------------------------- | -------------------------- | -------------------------------------- |
| Default colormap          | `viridis`                  | `parula` (`viridis` mapped internally) |
| Invalid colormap          | Falls back to `viridis`    | Falls back to `parula`                 |
| `electrode_style: "dots"` | Use `"dots"`               | Use `"on"`                             |
| `figure_size`             | Respected                  | Not used (fixed positions)             |
| `dpi`                     | Controls raster resolution | Not directly used                      |

## File Structure

```
eeg-topoplot-generator/
├── config.json                      # Shared configuration
├── generate_topography.py           # Python implementation (MNE)
├── generate_topography_eeglab.m     # MATLAB implementation (EEGLAB)
├── <your_data_file>.xlsx            # Input data (auto-detected)
├── topography_plots/                # Python output
└── topography_plots_eeglab/         # MATLAB output
```
