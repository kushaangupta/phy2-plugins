# Plugins to Phy2
These plugins add additional features to Phy2. Originally created for Phy1 by Peter Petersen and made compatible with Phy2 by Thomas Hainmueller. Newly updated implementations made by Mingze Dou.

## Features

### Newly Updated Features 
* **ImprovedISIAnalysis** (`alt+i`): Detects ISI conflicts using multiple metrics
* **StableMahalanobisDetection** (`alt+x`): Outlier detection with interactive visualization
* **ReclusterUMAP** (`alt+k`): Modern reclustering using UMAP and template matching
* **GoodLabelsPlugin**: Improved cluster organization that sorts by quality (good > mua > noise). Provides better workflow organization despite Phy's real-time update limitations.
* **AmplitudeHistogramPlugin**: Adds an amplitude-completeness diagnostic. Open `View > New view > AmplitudeHistogramView` to inspect full template-amplitude distributions and the estimated fraction of spikes missed below the detection threshold.

### Legacy Features
* **Reclustering** (`alt+shift+k`, `alt+shift+t`): KlustaKwik 2.0 based reclustering
* **Mahalanobis Distance** (`alt+shift+x`): Outlier detection (threshold: 16 std)
* **K-means Clustering** (`alt+shift+q`): Basic clustering with adjustable clusters
* **ISI Violation** (`alt+shift+i`): Visualize refractory period violations

## Dependencies
```bash
pip install pandas numpy scipy scikit-learn umap-learn
pip install --no-build-isolation klustakwik2
```

## Installation
1. Copy the plugin files from the `plugins` directory of this repository to your Phy user configuration directory, inside a `plugins` subfolder.
   - On Windows, this is typically `C:\Users\<YourUserName>\.phy\plugins`.
   - If the `plugins` folder doesn't exist, you will need to create it.
2. Copy other supporting files (e.g., `phy_config.py`, `klustakwik.exe` if used) from the root of this repository to your Phy user configuration directory (e.g., `~/.phy/` or `C:\Users\<YourUserName>\.phy`).
   - For Linux/Mac, build the `KlustaKwik` binary from the source code provided in this repository and place it in the Phy user configuration directory.
      ```bash
      unzip klustakwik.zip
      cd klustakwik
      make clean
      make
      ```
3. Install dependencies (see [Dependencies](#dependencies) section above).

## Amplitude completeness

Phy already includes amplitude-versus-time plots and small side histograms. The
`AmplitudeHistogramPlugin` complements those views with the standard
symmetry-based `amplitude_cutoff` estimate used for electrophysiology quality
control. The same implementation is shared with `CellMetrics` and
`AutoQuality`, so interactive and automated results agree.

Treat this number as a review aid rather than ground truth. It assumes a
stationary, roughly symmetric amplitude distribution; electrode drift,
multimodal clusters, and waveform changes can invalidate that assumption. It
also measures estimated missed detections, not contamination from other units.

## Authors
- Original Phy1: Peter Petersen
- Phy2 compatibility: Thomas Hainmueller
- New implementations: Mingze Dou
- Original amplitude-histogram concept: Luke Shaheen

## How to Cite
[![DOI](https://zenodo.org/badge/126424002.svg)](https://zenodo.org/badge/latestdoi/126424002)
