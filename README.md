# **Super Dead Leaves** pattern generator

## Overview

The **Super Dead Leaves** (SDL) pattern is an extension of the Dead Leaves pattern [[1]](#1) commonly used for the evaluation of texture reproduction fidelity in cell-phone photography [[2]](#2) and other applications.
Instead of using overalapping circles, the SDL utilizes Johan Gielis' **superformula** [3] to generating complex random shapes that better resemble overlapping leaves.
The random nature of the shapes is intended to allow the use of the pattern in the training and testing of machine learning-based image post-processing algorithms, posing a more substantial challenge than the predictable circular shapes in the traditional pattern (which has found anyway valuable uses in the field [4]).

   <a id="1">[1]</a> Gielis, Johan. "A generic geometric transformation that unifies a wide range of natural and abstract shapes." American journal of botany 90, 333-338 (2003)

   <a id="2">[2]</a> IEEE 1858-2023 Standard for Camera Phone Image Quality, https://standards.ieee.org/ieee/1858/6931/

## Features

- **Procedural Texture Generation**: Create diverse leaf-like shapes using the superformula.
- **Customizable Parameters**: Adjust parameters to vary shapes, sizes, and randomness.
- **Advanced Visualization**: Generate high-resolution images for research purposes.
- **Open Source**: Available for collaboration and modification by the scientific community.

## Installation

To install the required Python packages, use the `requirements.txt` file:

```bash
pip install -r requirements.txt



## Clone and Execution

### Clone the Repository

To clone the repository and get started:

```bash
git clone https://github.com/yourusername/super-dead-leaves.git
cd super-dead-leaves
```

### Run the Jupyter Notebook

Open the `super_dead_leaves_generator.ipynb` notebook in Jupyter to start generating textures:

```bash
jupyter notebook super_dead_leaves_generator.ipynb
```

### Generate Textures

Follow the instructions in the notebook to adjust parameters and create custom leaf textures. The notebook includes examples and descriptions to guide you through the process.

## Running on Binder

You can run this project directly on Binder without installing anything locally. Click the badge below to launch the notebook:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yourusername/super-dead-leaves/main)

## License

This project is licensed under the Creative Commons Zero (CC0) copyleft license. You are free to use, modify, and distribute this work without any restrictions. See the [LICENSE](LICENSE) file for more details.

