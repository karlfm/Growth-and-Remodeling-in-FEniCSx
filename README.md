# Supplementary source code for the paper "Growth and Remodelling Package in FEniCSx"

This repository contains the supplementary source code for the paper "Growth and Remodelling Package in FEniCSx" by Karl Munthe, Henrik Finsberg, Samuel Wall and Joakim Sundnes published in the FEniCS 2024 conference proceedings.

This paper presents a comparison of different growth laws implemented in the FEniCSx. The paper is available at ... (to be added)

## Installation

To run this program you need to install dolfinx. Please see [dolfinx installation instructions](https://github.com/FEniCS/dolfinx?tab=readme-ov-file#installation) for details.

If you have dolfinx installed, you can install the required dependencies by running

```bash
python3 -m pip install . 
```
in the root directory of this repository.

To ease the process we also provide a docker image with dolfinx installed as well as the code for the paper. To run the docker image, first build the image

You can get this docker image by running 
```
docker pull ghcr.io/karlfm/Growth-and-Remodeling-in-FEniCSx:main
```

To start a container please run the following command in the root directory of this repository:


```bash
docker run -ti -v $(pwd):/home/shared -w /home/shared ghcr.io/karlfm/Growth-and-Remodeling-in-FEniCSx:main
```

This will also mount your current directory to the `/home/shared` directory in the container. You can then run the code in the container, and the results will be saved in your current directory.

## Reproducing the results

To reproduce the result in the paper you can run 
```
python3 reproduce_figures.py
```

to reproduce the plots in the paper. See the [`examples/`](examples) directory for more examples.


## Citing
If you use this code in your research, please cite the paper as follows:

```
TBD
```

## License
MIT