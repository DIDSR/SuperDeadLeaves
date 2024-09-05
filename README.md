# Super Dead Leaves pattern generator

## Overview

The **Super Dead Leaves** (SDL) pattern is an open-source extension of the Dead Leaves pattern [1,2] used for the evaluation of texture reproduction fidelity in cell-phone photography [3] and other applications.
Instead of using overalapping circles, the new SDL utilizes Johan Gielis' **superformula** [4] to generating complex random shapes that better resemble overlapping leaves.
The random nature of the shapes is intended to enhance the utility of the pattern for image quality evaluation [5] and for training and testing machine learning-based image post-processing algorithms [6], posing a more substantial challenge than the predictable circular shapes.
  
  - [1] Cao, Frédéric, Frédéric Guichard, and Hervé Hornung. "Dead leaves model for measuring texture quality on a digital camera." In SPIE Digital Photography VI, vol. 7537, pp. 126-133 (2010) (https://doi.org/10.1117/12.838902)
  - [2] Lee, A.B., Mumford, D. and Huang, J. "Occlusion Models for Natural Images: A Statistical Study of a Scale-Invariant Dead Leaves Model." International Journal of Computer Vision 41, pp. 35–59 (2001) (https://doi.org/10.1023/A:1011109015675)
  - [3] IEEE 1858-2023 Standard for Camera Phone Image Quality (https://standards.ieee.org/ieee/1858/6931/)
  - [4] Gielis, Johan. "A generic geometric transformation that unifies a wide range of natural and abstract shapes." American journal of botany 90, pp. 333-338 (2003) (https://doi.org/10.3732/ajb.90.3.333)
  - [5] Artmann, Uwe. "Image quality assessment using the dead leaves target: experience with the latest approach and further investigations." In SPIE Digital Photography XI, vol. 9404, pp. 130-144 (2015) (https://doi.org/10.1117/12.2079609)
  - [6] Achddou, Raphaël, Yann Gousseau, and Saïd Ladjal. "Fully synthetic training for image restoration tasks." Computer Vision and Image Understanding 233, 103723 (2023). (https://doi.org/10.1016/j.cviu.2023.103723)
    
## Code features

- _Procedural Texture Generation_: create unlimited leaf-like shapes using the superformula.
- _Customizable Parameters_: adjust parameters to vary the individual shapes and the complete SDL chart.

## Installation

To run the code, install the required Python packages listed in the `requirements.txt` file, and clone the repository:

```bash
pip install -r requirements.txt
git clone https://github.com/DIDSR/SuperDeadLeaves.git
cd SuperDeadLeaves/
```

### Run the Jupyter Notebook locally or on Binder

Open the `super_dead_leaves_generator.ipynb` notebook in Jupyter to start generating textures:

```bash
jupyter notebook super_dead_leaves_generator.ipynb
```


> [!NOTE]
> Option not yet available.

You can run this project directly on Binder without installing anything locally. Click the badge below to launch the notebook:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yourusername/super-dead-leaves/main)



![SDL](https://github.com/user-attachments/assets/d50800bc-ee72-4cde-9170-cdaa9cdc599d)


## Disclaimer

This software and documentation (the "Software") were developed at the **US Food and Drug Administration** (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.
