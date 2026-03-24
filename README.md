# Reproducing Analyses for: Neural similarity when viewing social interactions differs by age and relates to social behavior

This repository contains code to reproduce the primary analyses reported in the manuscript.

## Data Availability

The dataset used in this project is publicly available via OSF: https://osf.io/e467j

Please download the data from OSF and place it the appropriate directories before running any analyses (see below for notes on paths).

---

## Repository Overview

This repository includes code for two main analyses:

### 1. Group-Level Permutation Testing on ISC Maps

* Script: `ISC_permutation/ISC_group_perm.py`
* Description:
  Performs group-level permutation testing on intersubject correlation (ISC) maps to assess statistical significance between age groups.

### 2. Brain–Behavior Modeling

* Script: `behavioral_modeling/ISC_behavioral_models.py`
* Description:
  Models the relationship between neural similarity (by brain region) and two independent measures of theory of mind.
  
---

## Requirements

We provide a Conda environment file to ensure reproducibility.

### Create the environment

```bash
conda env create -f neural_sim.yaml
```

### Activate the environment

```bash
conda activate neural_sim
```

---

## Usage

After setting up the environment and downloading the dataset:

1. Update any file paths in the scripts to point to your local copy of the OSF data.
2. Run analyses as needed. For example:

```bash
python ISC_permutation/ISC_group_perm.py
```

Additional scripts for brain–behavior modeling can be run similarly.
```bash
python behavioral_modeling/ISC_behavioral_models.py
```

---

## Notes

* The code is designed to reproduce the analyses as reported in the manuscript and may not be optimized for general use.
* Paths to data are not hardcoded for portability—users will need to specify their local directory structure.
* Depending on your system, some analyses (e.g., permutation testing) may be computationally intensive.

---

## Citation

If you use this code or dataset, please cite the associated manuscript.

---

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
