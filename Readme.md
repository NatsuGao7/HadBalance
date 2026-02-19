# Hadwiger Shape Priors with Conflict-Aware Objective Balancing for Generalizable Biomedical Image Segmentation

This repository contains the official implementation of HadBalance, a plug-and-play geometry-prior module for biomedical image segmentation. It seamlessly injects Hadwiger shape constraints into the training process while adaptively resolving gradient conflicts.

---

## **Dataset:**

We conducted our experiments on the [CUBS](https://data.mendeley.com/datasets/m7ndn58sv6/1), [CVC-ClinicDB](https://www.kaggle.com/datasets/orvile/cvc-clinicdb), and [DRIONS-DB](https://www.idiap.ch/software/bob/docs/bob/bob.db.drionsdb/master/index.html) datasets.

---

## **Algorithm Description:**

[![PDF Preview](./hadwiger.png)](./hadwiger.pdf)
The implementation of the Hadwiger module is illustrated in the figure above. For detailed code, please refer to hadwiger.py.

[![PDF Preview](./coab.png)](./coab.pdf)
The Conflict-Aware Objective Balancing (COAB) mechanism consists of two core modules: PGP and AGB. Specifically, PGP is responsible for filtering out harmful conflicting gradients (determining the optimal direction), while AGB calculates the optimal fusion weights for the remaining safe directions (determining the step size). Working in tandem, they constitute the robust COAB framework. For more details, please refer to the source code pgp.py and agb.py.
