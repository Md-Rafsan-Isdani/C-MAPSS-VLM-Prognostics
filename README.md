# C-MAPSS-VLM-Prognostics
Reproducible code and experimental artifacts for VLM-assisted prognostic analysis on NASA C-MAPSS.
# C-MAPSS VLM-Assisted Prognostics

This repository contains the code and experimental artifacts for the study:

**“Reliability, Abstention and Operating-Regime Sensitivity of Vision-Language Model Evidence in Multimodal Prognostics: A Multi-Seed Study on NASA C-MAPSS.”**

## Overview

This project investigates whether qualitative visual evidence extracted by a vision-language model (VLM) from multivariate sensor plots can complement conventional numerical remaining useful life (RUL) prediction.

The framework combines:

* Eight numerical RUL prediction models
* Qwen2.5-VL-7B-Instruct for qualitative visual analysis
* Deterministic constrained fusion of numerical predictions and visual evidence
* Qwen2.5-7B-Instruct for evidence analysis
* Three independent random seeds
* All four NASA C-MAPSS subsets

The VLM does **not** directly predict numerical RUL. Instead, it provides structured qualitative evidence regarding degradation behaviour, operating regime, notable sensors, terminal-phase indications, confidence, and figure readability.

## Dataset

The experiments use the publicly available **NASA C-MAPSS** benchmark dataset.

The dataset itself is not included in this repository. Users should obtain it from the official NASA Prognostics Center of Excellence repository.

## Experimental Configuration

The study evaluates the following C-MAPSS subsets:

* FD001
* FD002
* FD003
* FD004

Three independent random seeds are used:


42
1729
191


## Reproducibility

The repository contains the implementation and experimental artifacts required to reproduce the analyses reported in the manuscript.

The numerical branch uses deterministic fusion rules, while the VLM is used as a qualitative visual-evidence component.

## Citation

If you use this work in your research, please cite the associated publication.

