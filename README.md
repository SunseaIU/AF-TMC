# AF-TMC: Adaptive Feature-based Transition Probability Matrix Clustering

This repository contains the official MATLAB implementation of the **AF-TMC** algorithm for multi-view clustering. 

## 📁 Repository Structure

```text
├── core/                  # Core optimization steps
│   ├── updateAlpha.m      # View weight update
│   ├── updateF.m          # Embedding matrix update
│   ├── updateS.m          # Shared similarity matrix update
│   ├── updateTheta.m      # Transition matrix update
│   └── updateZ.m          # Representation matrix update
├── utils/                 # Evaluation and mathematical utility functions
│   ├── computeObjective.m # Calculates the objective function value
│   ├── evalClustering.m   # Evaluation metrics (ACC, NMI, ARI, F-score, etc.)
│   └── projSimplex.m      # Simplex projection utility
├── AF_TMC.m               # Main algorithm entry API
├── demo_main.m                 # Quick-start demonstration script (Recommended)
├── 3Sources.mat           # Sample multi-view dataset
└── README.md              # This documentation file