FROM ghcr.io/fenics/dolfinx/dolfinx:v0.9.0

COPY . /app
WORKDIR /app

RUN python3 -m pip install .