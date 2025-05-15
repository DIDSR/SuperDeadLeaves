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
# To avoid artifacts at the top and bottom poles (at polar angle 0 and 180, where all azimuthal values are valid 
# in a single point) the radii are multiplied by a squared sinusoidal weighting factor that forces a smooth 
# transition towards the poles (the radius at the equator is r_phi[phi]*r_theta[90], while at the top pole is 
# r_theta[0]*mean(r_phi)).
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
from scipy.stats import pareto
from skimage.transform import downscale_local_mean, rescale   # Library only necessary if I want to downsample the sampled pattern

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
                 a_range=[1.0, 1.0], b_range=[1.0, 1.0],
                 m_range1=[3, 8], m_range2=[3, 8], n1_range=[1.0, 5.0],
                 n2_range=[1.0, 5.0], n3_range=[1.0, 5.0],
                 rmin=0.015, rmax=0.15, PowerLawExp=3, num_shape_samples=37, spheric_target_r=-1, spheric_target_coord=None):
        """
        Initialize the SuperDeadLeaves3D generator.

        Parameters
        ----------
        vol_size : list of int
            Dimensions of the 3D volume [L, M, N].
        seed : int or None
            Random seed for reproducibility (None for random initialization).
        a_range, b_range : list
            Ranges for a and b parameters of the superformula. If not equal to 1, the resulting shapes might have discontinuities.
        m_range1, m_range2 : list
            Range for m parameter (number of lobes) in the two sampled superformulas. Use [2,2] for circular blobs, [0,0] for spheres. The first superformula defines the polar profile, and the second the azimuthal profile.
        n1_range, n2_range, n3_range : list
            Ranges for n1, n2, n3 parameters (safe to use negative values).
        rmin, rmax : float
            Minimum and maximum shape radius for the inverse power law sampling. A value of 1 corresponds to the full width of the volume (X axis).
        PowerLawExp : float
            Exponent of the (inverse) power law probability distribution (prob. ~ 1/f^x) used to sample the radii of the shapes (3 gives scale-invariant patterns in 2D).
        num_shape_samples : int
            Number of points to sample (using the Fibonacci uniform sphere sampling algorithm) to estimate each shape size. Fewer points will cause more variability in the estimation and the final size distribution might be a less perfect power law. But more points slow down the execution. Use few points for shapes that don't have a lot of irregularities (like blobs), and a single point for spheres.
        spheric_target_r, spheric_target_coord: float and list
            If radius > 0, insert a sphere of the given radius at the input relative coordinates (in units of a fraction of the X axis size). Using random coordinates if none provided. This option is useful for detection and shape discrimination tasks.
        """

               
        if len(vol_size) == 2:
            self.vol_size = [vol_size[0], vol_size[1], 1]   # Allow a 2D input for the 3D model
        else:
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

        # Initialize superformula parameters as class members
        self.a  = [a_range[0],  a_range[0]]
        self.b  = [b_range[0],  b_range[0]]
        self.m  = [m_range1[0], m_range2[1]]
        self.n1 = [n1_range[0], n1_range[0]]
        self.n2 = [n2_range[0], n2_range[0]]
        self.n3 = [n3_range[0], n3_range[0]]
        self.theta0 = 0.0    # Random phase offset for the polar superformula profile, to avoid starting with a lobe on the pole
        
        # Power law shape distribution parameters:
        self.PowerLawExp = PowerLawExp
        self.rmin = rmin
        self.rmax = max(rmax, rmin*1.01)   # Make sure rmax > rmin, or we will get stuck in infinite loop

        # Option to insert a sphere of the given radius at the input relative coordinates (disabled if radius < 0):  !!sphere!!
        if spheric_target_r > 0:
            self.spheric_target_r = min(spheric_target_r, 0.5)
            self.spheric_target_coord = spheric_target_coord
        else:
            self.spheric_target_coord = None
            self.spheric_target_r = -1.0
            
        # Pre-compute uniform points on a sphere (using the Fibonacci sphere algorithm) to estimate the shape size:
        indices = np.arange(0, num_shape_samples, dtype=float) + 0.5
        self.theta_Fibonacci = np.arccos(1 - 2*indices/num_shape_samples)
        self.phi_Fibonacci   = np.pi * (1 + 5**0.5) * indices
        


    def gielis_superformula(self, theta, phi):
        """
        Compute the 3D Gielis superformula radius for the two given pair of angles theta and phi.
        Calculation performed in float32 and using np.errstate to handle divide-by-zero and other invalid operations.
        The first superformula defines the polar profile, and the second the azimuthal profile.
        If m==0, then the output is a sphere and we simply return 1 for each angle.

        Parameters
        ----------
        theta, phi : ndarray
            Pairs of polar and azimuthal angle arrays for the two superformula shapes.

        Returns
        -------
        r_theta, r_phi : ndarray
            Computed radius for the two separate superformulas.
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


    def spherical_coordinates(self, X, Y, Z=None):
        """
        Convert the input Cartesian coordinate arrays (X,Y,Z) to spherical coordinate arrays: (r, θ, φ).
        Handle faster 2D case where Z is not input (assumed all zeros: theta=pi/2). 
        Compute theta with arctan2 instead of arccos for efficiency and numerical stability (avoid division by r).
        """    
        if Z is None:
            r = np.hypot(X, Y).astype(np.float32)
            theta = np.full_like(r, 0.5*np.pi, dtype=np.float32)
            phi = np.arctan2(Y, X).astype(np.float32)
        else:
            r = np.sqrt(X**2 + Y**2 + Z**2).astype(np.float32)
            theta = np.arctan2(np.hypot(X, Y), Z).astype(np.float32)
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
        self.theta0 = self.rng.uniform(0, np.pi)

        if self.vol_size[2] == 1:   # Requesting a 2D pattern
            self.m[0] = 0.0         #   Use a simple circle for the unused polar profile            
            
        # If requested, insert a sphere inside the pattern:   !!sphere!!
        if self.spheric_target_r > 0:
            self.m = [0, 0]
            
            if self.m_range1[1]<2 and self.m_range2[1]<2:   # Exception: if the pattern contains only spheres, then insert hexagonal shape
                self.m, self.n1, self.n2, self.n3 = [6, 6], [2.5, 2.5], [1, 1], [1, 1]   # Hexagon
                #self.m, self.n1, self.n2, self.n3 = [6, 6], [1000, 1000], [250, 250], [250, 250]   # Octagon
                #self.m, self.n1, self.n2, self.n3 = [9, 9], [   9,    9], [ 25,  25], [ 25,  25]   # Starfish shape
                print(f"     Inserting an hexagonal target among spheres: m={self.m[0]}, n1={self.n1[0]}, n2={self.n2[0]}, n3={self.n3[0]}")

            if self.spheric_target_coord is None:
                # Sample a random location for the inserted sphere inside the bounding box
                self.spheric_target_coord = [self.rng.uniform(self.spheric_target_r, 1-self.spheric_target_r), self.rng.uniform(self.spheric_target_r, 1-self.spheric_target_r), self.rng.uniform(self.spheric_target_r, 1-self.spheric_target_r)]


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
        r_theta, r_phi = self.gielis_superformula(self.theta_Fibonacci + self.theta0, self.phi_Fibonacci)

        # Combine the two radii with a sinusoidal squared weighting factor for a smooth transition towards the pole, where we use the mean value of r_theta (otherwise all azimuthal values are possible at polar angle = 0 or pi):
        weight = np.sin(self.theta_Fibonacci)**2
        mean_r_phi = np.mean(r_phi)
        radii = weight*r_theta*r_phi + (1-weight)*r_theta*mean_r_phi 
          # PREVIOUS VERSION: radii = r_theta * r_phi

        if visualize:
            import matplotlib.pyplot as plt
            print(f"     [estimate_shape_size] 3D superformula parameters: a={self.a[0]:.2f}, {self.a[1]:.2f}; b={self.b[0]:.2f}, {self.b[1]:.2f}; m={self.m[0]}, {self.m[1]}; n1={self.n1[0]:.2f}, {self.n1[1]:.2f}; n2={self.n2[0]:.2f}, {self.n2[1]:.2f}; n3={self.n3[0]:.2f}, {self.n3[1]:.2f}")
            print(f"                           Shape radius sampled with {len(radii)} Fibonacci points: mean={np.mean(radii)}, min={np.min(radii)}, max={np.max(radii)}")
            x = radii * np.sin(self.theta_Fibonacci) * np.cos(self.phi_Fibonacci)
            y = radii * np.sin(self.theta_Fibonacci) * np.sin(self.phi_Fibonacci)
            z = radii * np.cos(self.theta_Fibonacci)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')            
            ax.scatter(x, y, z, c=np.arange(0, len(x)), cmap='viridis', marker='.', s=20)  # 'o'
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()

        return np.min(radii), np.max(radii), np.mean(radii)


    def __random_quaternion_rotation(self, Xc, Yc, Zc, u):
        """
        Rotate the input cartesian coordinate arrays (Xc, Yc, Zc) using Ken Shoemake's quaternion-based uniform rotation method. 
        The 3D random rotation is defined by the input array with 3 uniform random numbers.
        This private function is used to randomize the orientation of the added shapes in 3D.
        """
        w = np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1])
        x = np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1])
        y = np.sqrt(u[0]) * np.sin(2 * np.pi * u[2])
        z = np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])

        # Directly apply quaternion rotation to Xc, Yc, Zc
        Xc_rot = (1 - 2*(y**2 + z**2)) * Xc + (2*(x*y - z*w)) * Yc + (2*(x*z + y*w)) * Zc
        Yc_rot = (2*(x*y + z*w)) * Xc + (1 - 2*(x**2 + z**2)) * Yc + (2*(y*z - x*w)) * Zc
        Zc_rot = (2*(x*z - y*w)) * Xc + (2*(y*z + x*w)) * Yc + (1 - 2*(x**2 + y**2)) * Zc

        return Xc_rot, Yc_rot, Zc_rot
        

    def add_shape(self, volume, shape_number, scaling_factor, center_x, center_y, center_z, max_radius, overlap_factor, nucleus_fraction):
        """
        Add a newly sampled 3D shape to the SDL3D volume, with the existing shapes occluding the new ones.
        The algorithm starts by defining a bounding box for the new shape (based on the estimated max radius) centered at the shape origin.
        The coordinates of the centers of every voxel in the bounding box are then  transformed from Cartesian to spherical coordinates. 
        A random rotation is applied to randomize the shape orientation. Then, the polar (theta) and azimuthal (phi) angles are fed to 
        the superformula to compute the radius of the shape in that voxel. 
        After rescaling the shape radius with the sampled scaling factor, we detect the voxels covered by the shape by comparing each 
        voxel radius to the shape radius. This calculation is performed using efficient numpy array slicing.
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
            If <= 0: any overlapping is allowed. If >0: factor to rescale the shape max_radius to create an exclusion zone with no overlap between shapes (1.0 allows shapes to touch, without overlapping).
        nucleus_fraction : float
            If > 0: create a central replica of the shape (of size radius*nucleus_fraction) that resembles a cell nucleus in a cytology image.
        """
        eps = 1.0e-20

        # RECTANGULAR EXCLUSION ZONE (much faster than scaling the actual shape):
        if overlap_factor > eps:
            self.overlap_factor = overlap_factor   # Save input parameter in the class
            # Check if there is any shape already inside a RECTANGULAR EXCLUSION ZONE around the new shape center:            
            x_min, x_max = max(0, int(center_x - max_radius * scaling_factor * overlap_factor)), max(min(self.vol_size[0], int(center_x + max_radius * scaling_factor * overlap_factor) + 1), 0)
            y_min, y_max = max(0, int(center_y - max_radius * scaling_factor * overlap_factor)), max(min(self.vol_size[1], int(center_y + max_radius * scaling_factor * overlap_factor) + 1), 0)
            z_min, z_max = max(0, int(center_z - max_radius * scaling_factor * overlap_factor)), max(min(self.vol_size[2], int(center_z + max_radius * scaling_factor * overlap_factor) + 1), 0)
            if (volume[x_min:x_max, y_min:y_max, z_min:z_max]>0).any():
                return   # Reject shape!


        # Define the bounding box volume-of-interest (VOI) for the shape, adding a margin to account for underestimation of the max_radius
        margin = 1.10
        x_min, x_max = max(0, int(center_x - max_radius * scaling_factor * margin)), max(min(self.vol_size[0], int(center_x + max_radius * scaling_factor * margin) + 1), 0)
        y_min, y_max = max(0, int(center_y - max_radius * scaling_factor * margin)), max(min(self.vol_size[1], int(center_y + max_radius * scaling_factor * margin) + 1), 0)
        z_min, z_max = max(0, int(center_z - max_radius * scaling_factor * margin)), max(min(self.vol_size[2], int(center_z + max_radius * scaling_factor * margin) + 1), 0)

        # Create a meshgrid for the bounding box: we will check if the center of each voxel is inside or outside the shape:
        X, Y, Z = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), np.arange(z_min, z_max), indexing='ij')

        # Create a mask with voxels that are not yet covered and potentially covered by the new shape (ie, value==0):
        potential_shape_mask = (volume[x_min:x_max, y_min:y_max, z_min:z_max] == 0)  # Select only uncovered voxels  !!WARNING!! might be comparing a float to 0 (to avoid casting uint16)
        if not potential_shape_mask.any():
            # Reject the shape at once if zero voxels are available at that location, for that max_size:
            return
        Xc = (X[potential_shape_mask] + 0.5 - center_x).astype(np.float32)  # This is now a 1D array with ONLY the voxels that not yet covered (minimizing calls to gielis_superformula)
        Yc = (Y[potential_shape_mask] + 0.5 - center_y).astype(np.float32)
        Zc = (Z[potential_shape_mask] + 0.5 - center_z).astype(np.float32)

        # Apply a random rotation to randomize the orientation of the new shape, and convert to spherical coordinates:
        if self.vol_size[2] > 1:  # 3D pattern: rotate cartesian coordinates
            Xc, Yc, Zc = self.__random_quaternion_rotation(Xc, Yc, Zc, self.rng.random(3))
            r_voxel, theta, phi = self.spherical_coordinates(Xc, Yc, Zc)
            
        else:                     # 2D pattern: randomize azimuthal angle only (no polar rotation)
            r_voxel, theta, phi = self.spherical_coordinates(Xc, Yc)  # Default Z=zeros
            phi = phi + self.rng.uniform(0, 2*np.pi)

        # Compute Gielis superformula radii ONLY for the potentially relevant voxels. 
        # Apply random offset to theta, since we only use the superformula profile for a span of 180 degrees.
        r_theta, r_phi = self.gielis_superformula(theta + self.theta0, phi)

        # Combine the two radii with a sinusoidal squared weighting factor for a smooth transition towards the pole, where we use the mean value of r_theta (otherwise all azimuthal values are possible at polar angle = 0 or pi):
        weight = np.sin(theta)**2
        mean_r_phi = np.mean(r_phi)
        R_gielis = ( weight*r_theta*r_phi + (1-weight)*r_theta*mean_r_phi ) * scaling_factor
          # Previous version: R_gielis = (r_theta * r_phi) * scaling_factor


        # Determine which of the pre-filtered voxels are actually inside the shape.
        inside_mask_filtered = (r_voxel < R_gielis)

        # Create a mask to update the original volume, combining pre-filtering covered voxels and Gielis-based filtering with the new shape:
        inside_mask = np.zeros_like(volume[x_min:x_max, y_min:y_max, z_min:z_max], dtype=bool)
        inside_mask[potential_shape_mask] = inside_mask_filtered   # This step puts the reduced values (1D) in the proper location inside the bigger volume

        # Fill the voxels inside the shape
        if volume.dtype == np.uint16:
            
            #volume[x_min:x_max, y_min:y_max, z_min:z_max][inside_mask] = min(shape_number, np.iinfo(np.uint16).max)   # avoid overflow when adding more than 65535 shapes            
            volume[x_min:x_max, y_min:y_max, z_min:z_max][inside_mask] = self.rng.integers(100,200)  # !!DeBuG!! LUNG: if reporting integers, sample a shape value in the interval

        else:
            color = self.rng.random()
            volume[x_min:x_max, y_min:y_max, z_min:z_max][inside_mask] = color

            # If the user chose to insert a sphere as the first shape, report the parameters and disable the flag:
            if self.spheric_target_r > 0:
                print(f"\n     Inserting a spheric target with color {color:.6f} and radius {scaling_factor:.2f} at [{center_x:.1f}, {center_y:.1f}, {center_z:.1f}] (distance in units of pixels).\n")
                self.spheric_target_r = -self.spheric_target_r   # Only insert the sphere once (sign reload later for multiple images)

        if nucleus_fraction > eps:
            # Add a central nucleus to the shape:
            inside_mask_filtered = (r_voxel < R_gielis*nucleus_fraction)
            #inside_mask = np.zeros_like(volume[x_min:x_max, y_min:y_max, z_min:z_max], dtype=bool)
            inside_mask[potential_shape_mask] = inside_mask_filtered
            if volume.dtype == np.uint16:

                # volume[x_min:x_max, y_min:y_max, z_min:z_max][inside_mask] //= 2
                volume[x_min:x_max, y_min:y_max, z_min:z_max][inside_mask] = 1       # !!DeBuG!! LUNG: set the interior of the shapes to value 1 (0 means uncovered voxel)

            else:
                volume[x_min:x_max, y_min:y_max, z_min:z_max][inside_mask] /= 2



    def generate(self, max_shapes=65500, enumerate_shapes=False, overlap_factor=0.0, nucleus_fraction=0.0, verbose=True):
        """
        Generate a 3D volume filled with superformula shapes until no empty voxels remain or until max_shapes is reached.

        Parameters
        ----------
        max_shapes : int
            Maximum number of shapes to generate.
        enumerate_shapes: bool
            If True, each shape has a unique integer value (uint16 volume); if false, then uniform random floats in the range [0, 1] are assigned to each shape (float32 volume).
        overlap_factor : float
            If <= 0: any overlapping is allowed. If >0: factor to rescale the shape max_radius to create an exclusion zone with no overlap between shapes (1.0 allows shapes to touch, without overlapping).
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

        # Progress report interval:
        verbose_interval = 500

        # Parameters for faster and early termination:
        speedup_threshold = 8.0/100.0   # Increase rmin when less than this fraction of voxels are uncovered to speedup the ending
        no_progress_limit = verbose_interval*1.5   # Stop after this number of consecutive unsuccessful shape insertion attempts

        total_voxels = np.prod(self.vol_size)
        empty_voxels = total_voxels
        no_progress_count = 0
        rmin = self.rmin

        self.final_shapes = -1
        time0 = time.time()
        time1 = time0
        for shape_number in range(1, max_shapes + 1):
            if empty_voxels == 0:
                self.final_shapes = shape_number
                print(f"\n ...All voxels in the volume are covered after sampling {shape_number} shapes...\n")
                break

            # Sample parameters for the next shape
            self.sample_superformula_params()

            # Estimate the shape size to be able to scale it to follow an approximate power-law size distribution  
            min_radius, max_radius, mean_radius = self.estimate_shape_size()

            # Sample shape center, including a margin for shapes starting outside the volume (in each side):           
            
            if volume.dtype == np.uint16:
                out_fraction = -1 * min(5*self.rmin, self.rmax/2)  # !!DeBuG!! LUNG: defining a negative out_fraction means that the centers are sampled only certain distance INSIDE the bbox!
            else:
                out_fraction = min(3*self.rmin, self.rmax/2)
                
            cx = self.rng.uniform(-self.vol_size[0]*out_fraction, (self.vol_size[0] - 1) + self.vol_size[0]*out_fraction)
            cy = self.rng.uniform(-self.vol_size[1]*out_fraction, (self.vol_size[1] - 1) + self.vol_size[1]*out_fraction)
            cz = self.rng.uniform(-self.vol_size[2]*out_fraction, (self.vol_size[2] - 1) + self.vol_size[2]*out_fraction)
           
            # Sample the radius from an inverse power-law distribution (ie, Pareto dist with power exponent - 1) with the input min and max size:
            r = np.inf
            while r>self.rmax:
                r = pareto.rvs(b=(self.PowerLawExp-1), scale=rmin, random_state=self.rng)
          
            # If requested, insert a sphere inside the pattern:   !!sphere!!
            if self.spheric_target_r > 0: 
                cx = self.vol_size[0] * self.spheric_target_coord[0]
                cy = self.vol_size[1] * self.spheric_target_coord[1]
                cz = self.vol_size[2] * self.spheric_target_coord[2]
                r  = self.spheric_target_r

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
                time_now = time.time()
                print(f"     Shape {shape_number}: Filled {total_voxels - empty_voxels}/{total_voxels} voxels = {100*(total_voxels-empty_voxels)/total_voxels:6.3f}%. Runtime: {time_now-time0:.2f} s = {(time_now-time0)/60:.1f} min (interval: {time_now-time1:.2f}).")
                if (time_now-time1) < 10:
                    verbose_interval = 2*verbose_interval   # Double the interval between reports if it takes less than 10 seconds
                time1 = time_now
                    
                # Speedup the ending when only few sparse holes remain, by generating larger shapes (mostly covered by previous shapes):
                if empty_voxels < (total_voxels * speedup_threshold):
                    rmin = min(rmin*2, self.rmax/2)
                    speedup_threshold = speedup_threshold/2
                    
            # Check if no progress was made (to stop early)
            if before_empty == after_empty:
                no_progress_count += 1
            else:
                no_progress_count = no_progress_count//2

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

        if self.spheric_target_coord is not None:
            self.spheric_target_r = abs(self.spheric_target_r)   # Restore the input value for multiple patterns
            self.spheric_target_coord = None                     # A random location will be sampled for any following pattern
        
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


    def borderFree(volume):
        """
        Processes a 3D numpy array volume to eliminate (set to 0) the voxels of any
        shape that touches the boundary of the volume.
        This auxiliar funcion is mostly useful for making nice volume rendering without shapes
        cut in half by the bounding box. It might also help to tile volumes without discontinuities 
        (but with a gap between tiles).
        The function modifies the input array in place. If you want to keep the original volume, send a copy 
        to the function and asign the return value to a new variable:  volume0 = borderFree(volume.copy())  

        Parameters:
        - volume (np.ndarray): A 3D numpy array representing the volume. The voxel values will be modified in place.

        Return (optional):
        - np.ndarray: The processed volume with modified voxel values.
        """
        # Define the dimensions of the volume
        depth, height, width = volume.shape
        
        # Check for surface voxels on each face of the volume
        for z in range(depth):  # Front and back faces
            for y in range(height):
                for x in range(width):
                    # Surface check for each face, except 2D cases (every voxel touches a surface in that case)
                    if (depth  > 2 and (z == 0 or z == depth-1) or 
                        height > 2 and (y == 0 or y == height-1) or 
                        width  > 2 and (x == 0 or x == width-1)):
                        
                        # Set to 0 the voxels of any shape touching the surface. 
                        # This code is absurdly inefficient because it checks every voxel in the entire volume for each shape!
                        if volume[z, y, x] > 0:
                            volume[volume == volume[z, y, x]] = 0
        
        return volume
    
 
    def downsample_volume(volume_in, downsample_factor=4, offset_zeros=True, local_mean=False):
        """
        Downsample the input volume by an integer factor using local mean pooling (skimage.transform.downscale_local_mean).   
        If the input has only 2 dimensions or a 3rd dimension of size 1, downsampling is applied only to the first 2 dimensions.

        Parameters:
            volume_in (numpy.ndarray): The input 2D or 3D volume.
            downsample_factor (int or tuple): Integer downsampling factor for all axis (or first two if 2D input).
            offset_zeros (bool): If True, set any empty voxel with value 0 to the expected mean value of 0.5.

        Returns:
            numpy.ndarray: The downsampled volume.
        """
        if (not isinstance(downsample_factor, int)) or (volume_in.shape[0] % downsample_factor != 0) or (volume_in.shape[1] % downsample_factor != 0):
            raise ValueError(f"!!Downsampling error!! Input volume dimensions must be divisible by the integer scale factor {downsample_factor}: volume_in.shape={volume_in.shape}")
        
        if volume_in.ndim > 2:
            if volume_in.shape[2] < downsample_factor:
                down_factor_tuple = (downsample_factor, downsample_factor, 1)
                down_scale_tuple = (1.0/downsample_factor, 1.0/downsample_factor, 1)
            else:
                down_factor_tuple = (downsample_factor, downsample_factor, downsample_factor)  # 3D input
                down_scale_tuple = (1.0/downsample_factor, 1.0/downsample_factor, 1.0/downsample_factor)
        else:
            down_factor_tuple = (downsample_factor, downsample_factor)  # 2D input
            down_scale_tuple = (1.0/downsample_factor, 1.0/downsample_factor)
        
        # If requested, replace all the empty voxels (value 0) with the value 0.5, to mitigate bias in the averaging (at extremely high resolutions it is likely that there will be holes in the pattern). Using a temporary copy of the volume instead of the original memory: 
        if offset_zeros:
            volume_in_tmp = volume_in.copy()
            volume_in_tmp[volume_in < 1e-10] = 0.5
        else:
            volume_in_tmp = volume_in


        # Apply skimage.transform local mean pooling for downsampling with antialiasing:
        
        if local_mean:
            volume_out = downscale_local_mean(volume_in_tmp, factors=down_factor_tuple)  # local mean pooling for downsampling
        else:
            volume_out = rescale(volume_in_tmp, scale=down_scale_tuple, order=1, anti_aliasing=True)  # Rescale with linear interpolation and anti-aliasing filtering

        return volume_out.astype(volume_in.dtype)   # Preserve original data type


#########################################################################################


if __name__ == "__main__":
    import matplotlib.pyplot as plt
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
