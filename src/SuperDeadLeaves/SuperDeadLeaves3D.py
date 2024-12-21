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
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto

class SuperDeadLeaves3D:
    """"
    SuperDeadLeaves3D is a class designed to generate a 3D volume filled with shapes based on the Gielis superformula.
    The complex stochastic patterns generated have a fractal, non-Gaussian structure. The design is inspired by the texture 
    reproduction fidelity chart Dead Leaves used for texture fidelity evaluation in photography. 
    The class provides methods to convert Cartesian coordinates to spherical coordinates, compute superformula radii 
    for given angles, and iteratively fill a 3D volume with these shapes until maximum shapes are reached
    or the volume is filled. This can be used for simulations, procedural generation, and modeling in a 3D space.
    """

    def __init__(self, vol_size=[150, 150, 150], seed=None,
                 a_range=[0.75, 1.25], b_range=[0.75, 1.25],
                 m_range1=[3, 8], m_range2=[3, 8], n1_range=[1.0, 5.0],
                 n2_range=[1.0, 5.0], n3_range=[1.0, 5.0], phase_range=[0.0, 360.0],
                 rmin=0.015, rmax=0.15, PowerLawExp=3, num_shape_samples=113):
        """
        Initialize the SuperDeadLeaves3D generator.

        Parameters
        ----------
        vol_size : list of int
            Dimensions of the 3D volume [L, M, N].
        seed : int or None
            Random seed for reproducibility (None for random initialization).
        a_range, b_range : list
            Ranges for a and b parameters of the superformula.
        m_range1, m_range2 : list
            Range for m parameter (number of lobes) in the two sampled superformulas. Use [2,2] for circles.
        n1_range, n2_range, n3_range : list
            Ranges for n1, n2, n3 parameters (safe to use negative values).
        phase_range : list
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
        self.n2 = [3.0, 4.0]
        self.n3 = [5.0, 6.0]
        self.phase = [0.0, 0.0]
        
        # Power law shape distribution parameters:
        self.PowerLawExp = PowerLawExp
        self.rmin = rmin
        self.rmax = max(rmax, rmin*1.01)   # Make sure rmax > rmin, or we will get stuck in infinite loop
            
        # Pre-compute uniform points on a sphere (using the Fibonacci sphere algorithm) to estimate the shape size:
        indices = np.arange(0, num_shape_samples, dtype=float) + 0.5
        self.phi_Fibonacci   = np.arccos(1 - 2*indices/num_shape_samples)
        self.theta_Fibonacci =  np.pi * (1 + 5**0.5) * indices
        
        

    def gielis_superformula(self, theta, phi):
        """
        Compute the 3D Gielis superformula radius for the two given pair of angles theta and phi.
        Calculation performed in float32 and using np.errstate to handle divide-by-zero and other invalid operations.

        Parameters
        ----------
        theta, phi : ndarray
            Pairs of azimuthal and polar angle coordinates for the two superformula shapes.

        Returns
        -------
        r_theta, r_phi : ndarray
            Computed radius for the two superformulas.
        """
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):

            if self.m[0] != 0:                
                r_theta = np.power(np.power(np.abs(np.cos(self.m[0] * theta/4.0).astype(np.float32) / self.a[0]), self.n2[0]) +
                          np.power(np.abs(np.sin(self.m[0] * theta/4.0).astype(np.float32) / self.b[0]), self.n3[0]), -1.0/self.n1[0]).astype(np.float32)
            else:
                r_theta = np.ones_like(theta).astype(np.float32)
                                        
            if self.m[1] != 0:                
                r_phi  = np.power(np.power(np.abs(np.cos(self.m[1] * phi/4.0).astype(np.float32) / self.a[1]), self.n2[1]) +
                         np.power(np.abs(np.sin(self.m[1] * phi/4.0).astype(np.float32) / self.b[1]), self.n3[1]), -1.0/self.n1[1]).astype(np.float32)
            else:
                r_phi  = np.ones_like(phi).astype(np.float32)

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

        # Optional code to prevent generating n1 exponents between -0.5 and 0.5 when range varies from negative to positive:
        if self.n1_range[0]<0.1 and self.n1_range[1]>-0.1:
            self.n1 = [np.where(np.abs(x) < 0.5, self.rng.uniform(1, 3), x) for x in self.n1]
        
        
    def estimate_shape_size(self, visualize=False):
        """
        Estimate the minimum, maximum, and average radius of the superformula shape by
        sampling the radius at many points uniformly distributed on the surface of a sphere.
        The angles of these points are precomputed at the class constructor.

        Parameters
        ----------
        visualize : bool
            If True, visualize the sampled points on the sphere using a 3D scatter plot, and report the superformula parameters.

        Returns
        -------
        min_radius, max_radius, mean_radius : float
            Minimum, maximum, and average radius found.
        """
        # Compute the superformula radii for points in many directions:
        r_theta, r_phi = self.gielis_superformula(self.theta_Fibonacci, self.phi_Fibonacci)
        radii = r_theta * r_phi

        if visualize:
            print(f"     Superformula parameter: a={self.a[0]:.2f}, {self.a[1]:.2f}; b={self.b[0]:.2f}, {self.b[1]:.2f}; m={self.m[0]}, {self.m[1]}; n1={self.n1[0]:.2f}, {self.n1[1]:.2f}; n2={self.n2[0]:.2f}, {self.n2[1]:.2f}; n3={self.n3[0]:.2f}, {self.n3[1]:.2f};")
            print(f"                             phase={np.degrees(self.phase[0]):.2f}, {np.degrees(self.phase[1]):.2f}; num_Fibonacci_samples={len(radii)}")
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


    def add_shape(self, volume, shape_number, scaling_factor, center_x, center_y, center_z, max_radius, overlap_factor, nucleus_fraction):
        """
        Add a newly sampled 3D shape to the SDL3D volume, with the existing shapes occluding the new ones.
        The algorithm starts by defining a bounding box for the new shape (based on the estimated max radius) centered at the shape origin.
        The coordinates of the centers of every voxel in the bounding box are then transformed from Cartesian to spherical coordinates,
        and their theta and phi angles are fed to the superformula (with a random phase shift) to compute the radius of the shape in that 
        voxel. After rescaling the shape radius with the sampled scaling factor, we detect the voxels covered by the shape 
        by comparing each voxel radius to the shape radius. This calculation is performed using efficient numpy array slicing.
        Finally, the voxels that are inside the shape and that are not covered (value still 0) are set to either the new shape number, if the volume 
        is of type 16-bit unsigned integer, or to a floating-point random value between 0 and 1 otherwise.

        Parameters
        ----------
        volume : numpy.ndarray
            3D volume to add the shape to. Can be of type np.uint16 or floating point.
            The function call modifies this input volume in place, it is not explicitly returned.
        shape_number : int
            Number of the shape to add.
        scaling_factor : float
            Scaling factor to apply to the shape, computed from the estimated mean radius and the radius sampled from the power-law distribution.
        center_x, center_y, center_z : float
            X, Y, and Z coordinates of the center of the shape.
        max_radius : float
            Maximum radius of the shape.
        overlap_factor : float
            If >0: factor to rescale the shape radius to create an exclusion zone with no overlap between shapes (a value of 1.0 allows shapes to touch).
        nucleus_fraction : float
            If > 0: create a central replica of the shape (of size radius*nucleus_fraction) that resembles a cell nucleus in a cytology image.
        """
        eps = 1.0e-40
        self.overlap_factor = overlap_factor   # Save input parameter in the class


        # Define the bounding box volume-of-interest (VOI) for the shape, adding a margin to account for underestimation of the max_radius
        margin = 1.10

        if overlap_factor > eps:            
            margin = max(margin, overlap_factor)   # Need to check voxels far from the shape for possible overlapping. WARNING: large margins will be slow to compute!!

        x_min, x_max = max(0, int(center_x - max_radius * scaling_factor * margin)), max(min(self.vol_size[0], int(center_x + max_radius * scaling_factor * margin) + 1), 0)
        y_min, y_max = max(0, int(center_y - max_radius * scaling_factor * margin)), max(min(self.vol_size[1], int(center_y + max_radius * scaling_factor * margin) + 1), 0)
        z_min, z_max = max(0, int(center_z - max_radius * scaling_factor * margin)), max(min(self.vol_size[2], int(center_z + max_radius * scaling_factor * margin) + 1), 0)

        # Create a meshgrid for the bounding box: we will check if the center of each voxel is inside or outside the shape:
        X, Y, Z = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), np.arange(z_min, z_max), indexing='ij')

        if overlap_factor < eps:
            # Create a mask with voxels that are not yet covered and potentially covered by the new shape (ie, value==0):
            potential_shape_mask = (volume[x_min:x_max, y_min:y_max, z_min:z_max] == 0)  # Select only uncovered voxels  !!WARNING!! might be comparing a float to 0 (to avoid casting uint16)
            Xc = (X[potential_shape_mask] + 0.5 - center_x).astype(np.float32)  # This is now a 1D array with ONLY the voxels that not yet covered (minimizing calls to gielis_superformula)
            Yc = (Y[potential_shape_mask] + 0.5 - center_y).astype(np.float32)
            Zc = (Z[potential_shape_mask] + 0.5 - center_z).astype(np.float32)
        else:
            # No masking can be used when we need to enforce empty space between shapes (all voxels equally important)
            Xc = (X + 0.5 - center_x).astype(np.float32)
            Yc = (Y + 0.5 - center_y).astype(np.float32)
            Zc = (Z + 0.5 - center_z).astype(np.float32)

        # Convert to voxel centers to spherical coordinates, and add the sampled phase to randomize the shape orientation:
        r_voxel, theta, phi = self.spherical_coordinates(Xc, Yc, Zc)
        theta += self.phase[0]
        phi   += self.phase[1]

        # Compute Gielis superformula radii ONLY for the potentially relevant voxels.
        r_theta, r_phi = self.gielis_superformula(theta, phi)
        R_gielis = (r_theta * r_phi) * scaling_factor

        if overlap_factor < eps:
            # Determine which of the pre-filtered voxels are actually inside the shape.
            inside_mask_filtered = (r_voxel <= R_gielis)

            # Create a mask to update the original volume, combining pre-filtering covered voxels and Gielis-based filtering with the new shape:
            inside_mask = np.zeros_like(volume[x_min:x_max, y_min:y_max, z_min:z_max], dtype=bool)
            inside_mask[potential_shape_mask] = inside_mask_filtered   # This step puts the reduced values (1D) in the proper location inside the bigger volume

        else:
            if ((r_voxel < (R_gielis*overlap_factor)) & (volume[x_min:x_max, y_min:y_max, z_min:z_max]>0)).any():
                # No overlap allowed within the exclusion zone: reject shape!
                return
            inside_mask = (r_voxel <= R_gielis)  # Select voxels inside the shape (already checked no overlapping above)

        # Fill the voxels inside the shape
        if volume.dtype == np.uint16:
            volume[x_min:x_max, y_min:y_max, z_min:z_max][inside_mask] = min(shape_number, np.iinfo(np.uint16).max)   # avoid overflow when adding more than 65535 shapes
        else:
            volume[x_min:x_max, y_min:y_max, z_min:z_max][inside_mask] = self.rng.random()

        if nucleus_fraction > eps:
            # Add a central nucleus to the shape:
            inside_mask = (r_voxel < R_gielis*nucleus_fraction) 
            volume[x_min:x_max, y_min:y_max, z_min:z_max][inside_mask] /= 2



    def generate(self, max_shapes=65500, enumerate_shapes=False, overlap_factor=0.0, nucleus_fraction=0.0, verbose=False):
        """
        Generate a 3D volume filled with superformula shapes until no empty voxels remain or until max_shapes is reached.

        Parameters
        ----------
        max_shapes : int
            Maximum number of shapes to generate.
        enumerate_shapes: bool
            If True, each shape has a unique integer value (uint16 volume); if false, then uniform random floats in the range [0, 1] are assigned to each shape (float32 volume).
        overlap_factor : float
            If >0: factor to rescale the shape radius to create an exclusion zone with no overlap between shapes (a value of 1.0 allows shapes to touch).
        nucleus_fraction : float
            If > 0: create a central replica of the shape (of size radius*nucleus_fraction) that resembles a cell nucleus in a cytology image (eg., 0.25).
        verbose : bool
            If True, print progress updates and accelerate the final part of the pattern generation.

        Returns
        -------
        volume : ndarray (np.uint16 or np.float32)
            Generated 3D volume with overlapping shapes. Uncovered voxels are set to 0. Covered voxels are set to the integer index of the shape that filled them for uint16 type, or a random number between 0 and 1 for float32 type.
        """
        # Declare the empty volume array:
        dtype = np.uint16 if enumerate_shapes else np.float32
        volume = np.zeros(self.vol_size, dtype=dtype)

        total_voxels = np.prod(self.vol_size)
        empty_voxels = total_voxels
        no_progress_count = 0
        no_progress_limit = 200
        speedup_threshold = 5.0/100.0  # Increase rmin when less than this fraction of voxels are uncovered, to speedup the ending
        rmin = self.rmin
        verbose_interval = 500

        self.final_shapes = -1
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
            self.add_shape(volume, shape_number, scaling_factor, cx, cy, cz, max_radius, overlap_factor, nucleus_fraction)

            # Count empty voxels before and after adding the shape
            before_empty = empty_voxels
            after_empty = np.count_nonzero(volume == 0)
            empty_voxels = after_empty

            # Verbose output and various tricks
            if verbose and shape_number % verbose_interval == 0:
                print(f"     Shape {shape_number}: Filled {total_voxels - empty_voxels}/{total_voxels} voxels = {100*(total_voxels-empty_voxels)/total_voxels:6.3f}%. Runtime: {time.time()-time0:.2f} s.")
                
                # Speedup the ending when only few sparse holes remain, by generating larger shapes (mostly covered by previous shapes):
                if empty_voxels < (total_voxels * speedup_threshold):
                    rmin = min(rmin*2, self.rmax/2)
                    speedup_threshold = speedup_threshold/2
                    
            # Check if no progress was made (to stop early)
            if before_empty == after_empty:
                no_progress_count += 1
            else:
                no_progress_count = no_progress_count//2    # = 0

            # Stop if no progress after many consecutive shapes
            if no_progress_count > no_progress_limit:
                self.final_shapes = shape_number
                print(f"\n ...stopping the generation after {no_progress_limit} consecutive shapes failed to cover any voxel (final_shapes={shape_number})...\n")
                break


        if enumerate_shapes and shape_number>np.iinfo(np.uint16).max:
            print(f" ... WARNING: {shape_number}>{np.iinfo(np.uint16).max}: to prevent overflow of the uint16 counters, all voxels covered by the final shapes share the max value of {np.iinfo(np.uint16).max}.")

        if self.final_shapes < 0:
                self.final_shapes = max_shapes
                print(f"\n ...Done sampling the {max_shapes} shapes...\n")

        return volume
    

    def report(self):
        """
        Report the value and type of the internal class variables (for long arrays, limit to the first elements).
        """
        print("\n   SuperDeadLeaves3D class instance variables:")
        attributes = vars(self)
        for attr_name, value in attributes.items():
            print(f"       - {attr_name} = {value[:4] if isinstance(value, (list, tuple, np.ndarray)) else value},\t type={type(value)}")
        print("\n")





#########################################################################################


if __name__ == "__main__":
    import tifffile

    # ** Example script to generate a 3D volume filled with random superformula shapes using the SuperDeadLeaves3D stochastic model.
 
    # Step 1: Define the volume size
    vol_size = [150, 150, 150]
    
    # Step 2: Initialize the SuperDeadLeaves3D generator
    seed = np.random.randint(1e4, 1e5)  # Use None for a random initialization
    print(f"\n ** SuperDeadLeaves3D generator initialized with seed={seed} and volume size={vol_size} **\n")
    
    SDL3D = SuperDeadLeaves3D(vol_size, seed=seed, m_range1=[2, 6], m_range2=[2, 4], n1_range=[2, 5], n2_range=[5, 9], n3_range=[2, 5], rmin=0.01)
    # Note: set m_range1 and m_range2 to [2,2] to generate circular blobs, or [0,0] for spheres
    SDL3D.report()

    # Step 3: Generate the volume
    max_shapes = 10000
    print(f"   Starting volume generation with up to {max_shapes} shapes...")
    start_time = time.time()  # Record start time    

    volume = SDL3D.generate(max_shapes=max_shapes, enumerate_shapes=False, verbose=True)  

    end_time = time.time()  # Record end time
    generation_time = end_time - start_time
    print(f"\n   Volume generation completed in {generation_time:.2f} sec ({SDL3D.final_shapes/generation_time:2e} shapes/sec).")

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
    print(  f"     Total Voxels: {total_voxels}")
    print(  f"     Filled Voxels: {filled_voxels}")
    print(  f"     Unique Shapes: {unique_shapes}")
    print(  f"     Percentage Filled: {filled_voxels / total_voxels * 100:.2f}%")
    
    # Step 6: Display the central Z slice
    mid_slice = vol_size[2] // 2  # Middle slice index along Z-axis
    print(f"\n   Displaying central slice Z={mid_slice}")
    plt.figure(figsize=(8, 8))
    plt.imshow(volume[:, :, mid_slice], cmap='nipy_spectral', origin='lower', interpolation='nearest')
    plt.title(f"Central Slice Z={mid_slice}")
    plt.colorbar(label="Shape Index")
    plt.tight_layout()
    plt.show()

    # Step 7: Display the projection of the volume along the Z axis
    if volume.shape[2] > 1:        
        plt.imshow(volume.mean(axis=2), cmap='bone', origin='lower', interpolation='bilinear')  # 'nipy_spectral'
        plt.title(f"Projection along Z-axis")
        plt.tight_layout()
        plt.show()

    # Step 8: show sample shapes
    SDL3D = SuperDeadLeaves3D(num_shape_samples=5000)
    SDL3D.sample_superformula_params()
    print("\n   Visualization of a 3D superformula shape, as it is sampled in the code to estimate its size:")
    print( f"     Estimated min, max and mean radii = {np.ravel(SDL3D.estimate_shape_size(visualize=True))}\n")
