# 
# **Super Dead Leaves-3D (SDL3D)**
# 
# The SDL3D pattern is a 3D extension of the Super Dead Leaves (SDL) pattern that creates voxelized volume
# completely filled with overlapping shapes randomly sampled using the superformula. This pattern is inspired
# by the texture reproduction fidelity chart Dead Leaves (DL) [1], defined in the international standards
# ISO-19567-2 [2] and IEEE-1858 [3]. 
# 
# In the SDL3D, the parameters of two independent superformula [4] 2D shapes are randomly sampled within
# the user-requested range of values. The superformula shapes are defined in polar coordinates.
# We create a 3D volume in spheric coordinates by utilizing the two 2D shapes as the theta and phi angles,
# and setting the shape radius in each direction as the multiplication of the radii of the two superformulas.
# Unlike in the SDL pattern, the SDL3D does not (yet) randomize the parameters separately in each lobe.
#
# The objective of the SDL3D pattern is to evaluate the performance of non-linear image processing algorithms
# and tomographic reconstruction algorithms based on machine-learning techniques, by using multiple
# realizations of these synthetic patterns generated at high-resolution and noise-free as a full-reference [5].
#
# ** References**
#  [1] Cao, Guichard and Hornung, "Dead leaves model for measuring texture quality on a digital camera" in SPIE Digital Photography VI, vol. 7537, p. 126-133 (2010)
#  [2] ISO/TS 19567-2:2019, "Photography — Digital cameras, Part 2: Texture analysis using stochastic pattern" (2019)
#  [3] IEEE 1858-2023, "IEEE Standard for Camera Phone Image Quality (CPIQ)" (2023) 
#  [4] Johan Gielis. "A generic geometric transformation that unifies a wide range of natural and abstract shapes." American journal of botany 90, p. 333-338 (2003)
#  [5] Kirk, L., Herzer, P., Artmann, U., and Kunz, D., "Description of texture loss using the dead leaves target: current issues and a new intrinsic approach." SPIE Digital Photography X, 9023, p. 112–120 (2014)
#
#
# -------------
# 
# **Author**: Andreu Badal (Andreu.Badal-Soler (at) fda.hhs.gov)
# 
# **Date**: 2024/12/18
# 
# **Disclaimer**
#    This software and documentation (the "Software") were developed at the US Food and Drug Administration
#    (FDA) by employees of the Federal Government in the course of their official duties.
#    Pursuant to Title 17, Section 105 of 
#    the United States Code, this work is not subject to copyright protection and is in the public domain.
#    Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to
#    deal in the Software without restriction, including without limitation the rights to use, copy,
#    modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and
#    to permit persons to whom the Software is furnished to do so.
#    FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code,
#    documentation or compiled executables, and makes no guarantees, expressed or implied, about its
#    quality, reliability, or any other characteristic. Further, use of this code in no way implies
#    endorsement by the FDA or confers any advantage in regulatory decisions. Although this software
#    can be redistributed and/or modified freely, we ask that any derivative works bear some notice that
#    they are derived from it, and any modified versions bear some notice that they have been modified.
#    
# -------------

#########################################################################################

import time
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto


class SuperDeadLeaves3D:
    def __init__(self, vol_size=[150, 150, 150], seed=None,
                 a_range=(0.5, 1.5), b_range=(0.5, 1.5),
                 m_range1=(3, 8), m_range2=(3, 8), n1_range=(0.5, 5.0),
                 n2_range=(0.5, 5.0), n3_range=(0.5, 5.0), phase_range=(0.0, 360.0),
                 rmin=0.015, rmax=0.15, PowerLawExp=3, num_shape_samples=113):
        """
        Initialize the SuperDeadLeaves3D generator.

        Parameters
        ----------
        vol_size : list of int
            Dimensions of the 3D volume [L, M, N].
        seed : int or None
            Random seed for reproducibility (None for random initialization).
        a_range, b_range : tuple
            Ranges for a and b parameters of the superformula.
        m_range1, m_range2 : tuple
            Range for m parameter (number of lobes) in the two sampled superformulas. Use [2,2] for circles.
        n1_range, n2_range, n3_range : tuple
            Ranges for n1, n2, n3 parameters.
        phase_range : tuple
            Range for phase shift (in degrees) to randomize the shape orientation in 3D.
        rmin, rmax : float
            Minimum and maximum shape radius for the inverse power law sampling. A value of 1 corresponds to the full width of the volume (X axis).
        PowerLawExp : float
            Exponent of the (inverse) power law probability distribution (prob. ~ 1/f^x) used to sample the radii of the shapes (3 gives scale-invariant patterns in 2D).
        """
        self.vol_size = vol_size
        self.final_shapes = 0
        self.rng = np.random.default_rng(seed)

        # Parameter ranges
        self.a_range = a_range
        self.b_range = b_range
        self.m_range1 = m_range1
        self.m_range2 = m_range2
        self.n1_range = n1_range
        self.n2_range = n2_range
        self.n3_range = n3_range
        self.phase_range = phase_range

        # Initialize superformula parameters as members
        self.a = [1.0, 1.0]
        self.b = [1.0, 1.0]
        self.m = [4.0, 6.0]
        self.n1 = [2.0, 2.0]
        self.n2 = [1.0, 1.0]
        self.n3 = [1.0, 1.0]
        self.phase = [0.0, 0.0]
        
        # Power law shape distribution parameters:
        self.PowerLawExp = PowerLawExp
        self.rmin = rmin
        self.rmax = rmax

        # Pre-compute uniform points on a sphere (using the Fibonacci sphere algorithm) to estimate the shape size:
        indices = np.arange(0, num_shape_samples, dtype=float) + 0.5
        self.phi_Fibonacci   = np.arccos(1 - 2*indices/num_shape_samples)
        self.theta_Fibonacci =  np.pi * (1 + 5**0.5) * indices
        
        

    def gielis_superformula(self, theta, phi):
        """
        Compute the 3D Gielis superformula radius for the two given pair of angles theta and phi.

        Parameters
        ----------
        theta, phi : ndarray
            Pairs of azimuthal and polar angle coordinates for the two superformula shapes.

        Returns
        -------
        r_theta, r_phi : ndarray
            Computed radius for the two superformulas.
        """
        r_theta = np.power(np.power(np.abs(np.cos(self.m[0] * theta/4.0).astype(np.float32) / self.a[0]), self.n2[0]) +
                           np.power(np.abs(np.sin(self.m[0] * theta/4.0).astype(np.float32) / self.b[0]), self.n3[0]), -1.0/self.n1[0]).astype(np.float32)

        r_phi  = np.power(np.power(np.abs(np.cos(self.m[1] * phi/4.0).astype(np.float32) / self.a[1]), self.n2[1]) +
                           np.power(np.abs(np.sin(self.m[1] * phi/4.0).astype(np.float32) / self.b[1]), self.n3[1]), -1.0/self.n1[1]).astype(np.float32)

        return r_theta, r_phi

    def spherical_coordinates(self, X, Y, Z):
        """
        Convert the input Cartesian coordinate arrays (x,y,z) to spherical coordinate arrays: r, θ, φ.
        θ = arccos(z/r), φ = atan2(y, x).
        """
        r = np.sqrt(X * X + Y * Y + Z * Z).astype(np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = np.where(r > 0, np.arccos(Z / r).astype(np.float32), 0.0)
        phi = np.arctan2(Y, X).astype(np.float32)
        return r, theta, phi

    def sample_superformula_params(self):
        """
        Sample two sets of random superformula parameters and store them as internal class members.
        """
        self.a  = [self.rng.uniform(*self.a_range),    self.rng.uniform(*self.a_range)]
        self.b  = [self.rng.uniform(*self.b_range),    self.rng.uniform(*self.b_range)]
        self.m  = [self.rng.integers(self.m_range1[0], self.m_range1[1]+1), self.rng.integers(self.m_range2[0], self.m_range2[1]+1)]
        self.n1 = [self.rng.uniform(*self.n1_range), self.rng.uniform(*self.n1_range)]        
        self.n2 = [self.rng.uniform(*self.n2_range), self.rng.uniform(*self.n2_range)]
        self.n3 = [self.rng.uniform(*self.n3_range), self.rng.uniform(*self.n3_range)]
        self.phase = [np.radians(self.rng.uniform(*self.phase_range)), np.radians(self.rng.uniform(*self.phase_range))]

        # Optional code to prevent generating exponents between -1 and 1 with range values from negative to positive:
        if self.n1_range[0]<0.1 and self.n1_range[1]>-0.1:
            self.n1 = [np.where(np.abs(x) < 1, self.rng.uniform(1, 3), x) for x in self.n1]
        if self.n2_range[0]<0.1 and self.n2_range[1]>-0.1:
            self.n2 = [np.where(np.abs(x) < 1, self.rng.uniform(1, 3), x) for x in self.n2]
        if self.n3_range[0]<0.1 and self.n3_range[1]>-0.1:
            self.n3 = [np.where(np.abs(x) < 1, self.rng.uniform(1, 3), x) for x in self.n3]
        
        
        
    def estimate_shape_size(self, visualize=False):
        """
        Estimate the minimum, maximum, and average radius of the superformula shape by
        sampling the radius at many points uniformly distributed on the surface of a sphere.
        The angles of these points are precomputed at the class constructor.

        Parameters
        ----------
        visualize : bool
            If True, visualize the sampled points on the sphere using a 3D scatter plot.

        Returns
        -------
        min_radius, max_radius, mean_radius : float
            Minimum, maximum, and average radius found.
        """
        # Compute the superformula radii for points in many directions:
        r_theta, r_phi = self.gielis_superformula(self.theta_Fibonacci, self.phi_Fibonacci)
        radii = r_theta * r_phi

        if visualize:
            x = radii * np.sin(self.phi_Fibonacci) * np.cos(self.theta_Fibonacci)
            y = radii * np.sin(self.phi_Fibonacci) * np.sin(self.theta_Fibonacci)
            z = radii * np.cos(self.phi_Fibonacci)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')            
            ax.scatter(x, y, z, c=np.arange(0, len(x)), cmap='viridis', marker='.', s=20)  # 'o'
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()

        return np.min(radii), np.max(radii), np.mean(radii)

        if visualize:
            x = radii * np.sin(self.phi_Fibonacci) * np.cos(self.theta_Fibonacci)
            y = radii * np.sin(self.phi_Fibonacci) * np.sin(self.theta_Fibonacci)
            z = radii * np.cos(self.phi_Fibonacci)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')            
            ax.scatter(x, y, z, c=np.arange(0, len(x)), cmap='viridis', marker='.', s=20)  # 'o'
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()

        return np.min(radii), np.max(radii), np.mean(radii)

    def add_shape(self, volume, shape_number, scaling_factor, center_x, center_y, center_z, max_radius):
        """
        Add a newly sampled 3D shape to the SDL3D volume, with the existing shapes occluding the new ones.
        The algorithm defines a bounding box for the new shape (based on the estimated max radius), centered at the shape origin. 
        The coordinates of the centers of every voxel are transformed from Cartesian to spherical coordinate system, and their 
        theta and phi angles are fed to the superformula (with a random phase shift) to compute the radius of the shape in that 
        direction. After rescaling the shape radius with the sampled scaling factor, we can detect the voxels covered by the 
        shape comparing each voxel radius to the shape radius. This calculation is performed using efficient numpy array slicing.
        Finally, the voxels that are inside the shape and that are not covered (value still 0) are set to the new shape number as 
        an unsigned integer, 16-bit (maximum value set to 65535).

        Parameters
        ----------
        volume : numpy.ndarray
            3D volume to add the shape to.
        shape_number : int
            Number of the shape to add.
        scaling_factor : float
            Scaling factor to apply to the shape, computed from the estimated mean radius and the radius sampled from the power-law distribution.
        center_x, center_y, center_z : float
            X, Y, and Z coordinates of the center of the shape.
        max_radius : float
            Maximum radius of the shape.
        """
        # Define the bounding box for the shape, adding a margin to the estimates max_radius
        margin = 1.15
        x_min, x_max = max(0, int(center_x - max_radius * scaling_factor * margin)), max(min(self.vol_size[0], int(center_x + max_radius * scaling_factor * margin) + 1), 0)  # 3)
        y_min, y_max = max(0, int(center_y - max_radius * scaling_factor * margin)), max(min(self.vol_size[1], int(center_y + max_radius * scaling_factor * margin) + 1), 0)  # 3)
        z_min, z_max = max(0, int(center_z - max_radius * scaling_factor * margin)), max(min(self.vol_size[2], int(center_z + max_radius * scaling_factor * margin) + 1), 0)  # 3)

        # Create a meshgrid for the bounding box
        x_range = np.arange(x_min, x_max)
        y_range = np.arange(y_min, y_max)
        z_range = np.arange(z_min, z_max)
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')

        # Convert to spherical coordinates
        Xc, Yc, Zc = (X - center_x).astype(np.float32), (Y - center_y).astype(np.float32), (Z - center_z).astype(np.float32)
        r, theta, phi = self.spherical_coordinates(Xc, Yc, Zc)

        # Apply phase shift to randomize the shape orientation in 3D
        theta += self.phase[0]
        phi   += self.phase[1]

        # Compute Gielis superformula radii
        r_theta, r_phi = self.gielis_superformula(theta, phi)
        R_gielis = (r_theta * r_phi) * scaling_factor

        # Determine which voxels are inside the shape and are still empty
        inside_mask = (r <= R_gielis) & (volume[x_min:x_max, y_min:y_max, z_min:z_max] == 0)

        # Fill the voxels inside the shape
        volume[x_min:x_max, y_min:y_max, z_min:z_max][inside_mask] = shape_number

        return volume

<<<<<<< HEAD
    def generate(self, max_shapes=65500, verbose=False, enumerate_shapes=True):
=======


    def generate(self, max_shapes=65500, verbose=False):
>>>>>>> eee20d76d6544ee96dcdead60ae01fcce4283155
        """
        Generate a 3D volume filled with superformula shapes until no empty voxels remain or until max_shapes is reached.

        Parameters
        ----------
        max_shapes : int
            Maximum number of shapes to generate.    
        verbose : bool
            If True, print progress updates and accelerate the final part of the pattern generation.
        enumerate_shapes: bool
            If True, each shape as a unique integer value, if false, then uniform floats in the range [0, 1]

        Returns
        -------
        volume : ndarray (np.uint16)
            Generated 3D volume pattern where each voxel is either 0 (empty) or contains the integer index of the shape that filled it.
        """
        # Declare the empty volume array:
        dtype = np.uint16 if enumerate_shapes else float
        volume = np.zeros(self.vol_size, dtype=dtype)

        total_voxels = np.prod(self.vol_size)
        empty_voxels = total_voxels
        no_progress_count = 0
        no_progress_limit = 500
        speedup_threshold = 6.0/100.0  # Increase rmin when less than this fraction of voxels are uncovered, to speedup the ending
        rmin = self.rmin
        verbose_interval = 500
        tmp_file = False  # Set to True to output the volume at 50% complete

        time0 = time.time()
        for shape_number in range(1, max_shapes + 1):
            if empty_voxels == 0:
                self.final_shapes = shape_number
                print(f"\n ...All voxels in the volume are covered after sampling {shape_number} shapes...\n")
                break

            # Sample parameters for the next shape
            self.sample_superformula_params()

            min_radius, max_radius, mean_radius = self.estimate_shape_size()

            # Sample shape center, including a margin for shapes starting outside the volume (in each side):           
            out_fraction = 4*self.rmin
            cx = self.rng.uniform(-self.vol_size[0]*out_fraction, (self.vol_size[0] - 1) + self.vol_size[0]*out_fraction)
            cy = self.rng.uniform(-self.vol_size[1]*out_fraction, (self.vol_size[1] - 1) + self.vol_size[1]*out_fraction)
            cz = self.rng.uniform(-self.vol_size[2]*out_fraction, (self.vol_size[2] - 1) + self.vol_size[2]*out_fraction)

            # Sample the radius from an inverse power-law distribution (ie, Pareto dist with power exponent - 1) with the input min and max size:
            r = np.inf
            while r>self.rmax:
                r = pareto.rvs(b=(self.PowerLawExp-1), scale=rmin, random_state=self.rng)

            # Scale the shape so that the mean radius is equal to the sampled fraction of the image (X axis width)
            scaling_factor = r * self.vol_size[0] / mean_radius  

            # Add the shape to the volume
            if enumerate_shapes:
                shape_value = min(shape_number, np.iinfo(np.uint16).max)
            else:
                shape_value = self.rng.random()
            volume = self.add_shape(volume, shape_value, scaling_factor, cx,
                                    cy, cz, max_radius)

            # Count empty voxels before and after adding the shape
            before_empty = empty_voxels
            after_empty = np.count_nonzero(volume == 0) # <-- check this line
            empty_voxels = after_empty

            # Verbose output and various tricks
            if verbose and shape_number % verbose_interval == 0:
                print(f"Shape {shape_number}: Filled {total_voxels - empty_voxels}/{total_voxels} voxels = {100*(total_voxels-empty_voxels)/total_voxels:6.3f}%. Runtime: {time.time()-time0:.2f} s.")

                # Speedup the ending when only few sparse holes remain, by generating larger shapes (mostly covered by previous shapes):
                if empty_voxels < (total_voxels * speedup_threshold):
                    rmin = min(rmin*2, self.rmax/2)
                    speedup_threshold = speedup_threshold/2

                if empty_voxels < total_voxels/2 and tmp_file is True:
                    tmp_file = False
                    tifffile.imwrite("tmp_50percent_complete.tif", volume.transpose(2,0,1)[np.newaxis, :, np.newaxis, :, :], imagej=True)

            # Check if no progress was made (to stop early)
            if before_empty == after_empty:
                no_progress_count += 1
            else:
                no_progress_count = 0

            # Stop if no progress after many consecutive shapes
            if no_progress_count > no_progress_limit:
                print(f"\n ...stopping the generation after {no_progress_limit} consecutive shapes failed to cover any voxel (final_shapes={shape_number})...\n")
                self.final_shapes = shape_number
                break

        if shape_number > np.iinfo(np.uint16).max:
            print(f" ... WARNING: {shape_number}>{np.iinfo(np.uint16).max}: to prevent overflow of the uint16 counters, all voxels covered by the final shapes share the max value of {np.iinfo(np.uint16).max}.")

        return volume

        # TODO: code function to convert the uint16 voxel values into random float32 values (0,1)
        # def convert_to_float32(volume):


#########################################################################################
if __name__ == "__main__":
    # ** Example script to generate a 3D volume filled with random superformula shapes using the SuperDeadLeaves3D stochastic model.

    # Step 1: Define the volume size
    vol_size = [500, 500, 500]   # [512, 512, 512]
    
    # Step 2: Initialize the SuperDeadLeaves3D generator
    seed = np.random.randint(1e4, 1e5)  # Use None for a random initialization
    print(f"\n ** SuperDeadLeaves3D generator initialized with seed={seed} **\n")
    print(f"   Volume size set to: {vol_size}")

    SDL3D = SuperDeadLeaves3D(vol_size, seed=seed, a_range=(1.0, 1.0), b_range=(1.0, 1.0), m_range1=(2, 6), m_range2=(2, 4), n1_range=(1, 4), n2_range=(5, 9), n3_range=(2, 5), rmin=0.02)
    
    # Note: set m_range1 and m_range2 to (2,2) to generate circular blobs.

    # Step 3: Generate the volume
    max_shapes = 5000 # 50000
    print(f"   Starting volume generation with up to {max_shapes} shapes...")
    start_time = time.time()  # Record start time    

    volume = SDL3D.generate(max_shapes=max_shapes, verbose=True)  

    end_time = time.time()  # Record end time
    generation_time = end_time - start_time
    print(f"   Volume generation completed in {generation_time:.2f} sec ({SDL3D.final_shapes/generation_time:2f} shapes/sec).")

    # Step 4: Export the volume as a multi-page TIFF file with the correct axis order for ImageJ (TZCYX)
    output_file = f"SuperDeadLeaves3D_Shapes_{seed}_shapes{SDL3D.final_shapes}_rmin{SDL3D.rmin}_{vol_size[0]}x{vol_size[1]}x{vol_size[2]}.tif"
    print(f"   Exporting the final volume to file: {output_file}")   
    tifffile.imwrite(output_file, volume.transpose(2,1,0))
    # Optional: export as a raw binary file
    # volume.tofile(output_file+".raw")    

    # Step 5: Report basic statistics about the volume
    filled_voxels = np.count_nonzero(volume)  # Count filled voxels
    unique_shapes = np.max(volume)  # Count unique shapes
    total_voxels  = np.prod(vol_size)  # Total voxels in the volume

    print(f"\n   Volume Statistics (volume.shape={volume.shape}):")
    print(f"     Total Voxels: {total_voxels}")
    print(f"     Filled Voxels: {filled_voxels}")
    print(f"     Unique Shapes: {unique_shapes}")
    print(f"     Percentage Filled: {filled_voxels / total_voxels * 100:.2f}%")
    
    # Step 6: Display the central Z slice
    mid_slice = vol_size[2] // 2  # Middle slice index along Z-axis
    print(f"   Displaying central slice Z={mid_slice}")
    plt.figure(figsize=(8, 8))
    plt.imshow(volume[:, :, mid_slice], cmap='nipy_spectral', origin='lower', interpolation='nearest')
    plt.title(f"Central Slice Z={mid_slice}")
    plt.colorbar(label="Shape Index")
    plt.tight_layout()
    plt.show()

    # Step 7: Display the projection of the volume along the X axis
    plt.imshow(volume.mean(axis=0), cmap='nipy_spectral', origin='lower', interpolation='bilinear') # 'nearest'
    plt.title(f"Projection along X-axis")
    plt.colorbar(label="Shape Index")
    plt.tight_layout()
    plt.show()

    # Step 8: show sample shapes
    SDL3D.sample_superformula_params()
    SDL3D.estimate_shape_size(visualize=True)

    SDL3D.sample_superformula_params()
    SDL3D.m = [9,9]
    SDL3D.estimate_shape_size(visualize=True)
