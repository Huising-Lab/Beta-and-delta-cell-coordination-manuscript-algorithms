# Heterogeneity in the coordination of delta cells with beta cells is driven by both paracrine signals and low-density Cx36 gap junctions
This repository contains the python scripts used in the paper: 

**Heterogeneity in the coordination of delta cells with beta cells is driven by both paracrine signals and low-density Cx36 gap junctions**

Authors: Mohammad S. Pourhosseinzadeh1, Jessica L. Huang1, Donghan Shin1, Ryan G. Hart1, Luhaiza Y.  Framroze1, Jaresley V. Guillen1, Joel Sanchez1, Ramir V. Tirado1, Kelechi Unanawa1, and Mark O. Huising1,2

1. Department of Neurobiology, Physiology, and Behavioral Biology, University of California Davis, Davis, CA 95616
2. Department of Physiology and Membrane Biology, University of California, Davis, CA, 95616

PNAS, September 15, 2025

DOI:[...](URL)

## Repository contents
- 'photon_counts_final.py' - Converts the raw data collected from a laser scanning confocal microscope into approximated photon counts
- 'time_series_filter_final.py' - Uses the photon counts data as input to filter data into low-pass, band-pass, and high-pass filtered data with annotations for the start/end of a peak and start/end of a plateau
- 'features_final.py' -Uses the identified start/end of peaks and start/end of plateau from above to derive many "features" including pearsons correlation, duty cycle, and on/off rate
- 'gap_junction_quantification_final.py' -Used in a semi-automated approach to identify gap junctions in stained images and assign them to the surface of beta or delta cells

## Getting started
All scripts were written in python and require no installation to run.

Python version 3.7.11

**Package --Version**

numpy --1.21.5

pandas --1.3.5

scipy --1.7.3

matplotlib --3.5.2

tifffile --2021.7.2

opencv-python --3.4.13.47
