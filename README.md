# evomap-package

Replication Code for the paper: 'evomap - a Toolbox for Dynamic Mapping in Python'.

First, install the required packages:

```
conda env create -f evomap-package-environment.yml
conda activate evomap-package
```

Then, run the replication code:

```
jupyter notebook src/evomap-demo.ipynb
```

A static version of this demo is available in 'src/evomap-demo.html'.

Note: Currently, evomap builds its C extensions upon installation on the system. Thus, it requires a C compiler to be present. The right C compiler depends upon your system, e.g. GCC on Linux or MSVC on Windows. For details, see the Cython documentation. 

Last updated/tested: May 13th using MacOS 13.5.2, conda 24.4.0, and pip 24.0