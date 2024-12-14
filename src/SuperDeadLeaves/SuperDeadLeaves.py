# 
# **Super Dead Leaves (SDL)**
# 
# The SDL pattern is an extension of the texture reproduction fidelity chart Dead Leaves (DL) [1] 
# defined in the international standards ISO-19567-2 [2] and IEEE-1858 [3]. 
# In the SDL, the overlapping random circles have been replaced by more complex shapes generated 
# using Johan Gielis\' _superformula_ [4], a generalization of the superellipse formula that can
# generate a wide variety of geometric shapes with multiple lobes that resemble real leaves. 
# To generate the pattern, a large number of shapes are sampled and "stacked" underneath the
# previously generated shapes until all pixels are covered.
#
# To further increase the variety and unpredictability of the shapes, the _superformula_ parameters 
# in the SDL can be randomly sampled separately in each lobe to generate even more organic-looking shapes. 
# Following the Dead Leaves model, the area of each shape matches the area of a circle with a radius
# sampled from a power law distribution with exponent -3 (scale invariant). The center of the shapes 
# are sampled with a uniform distribution on the image plane (the center can be located outside the image).
# 
# The superformula in polar coordinates is given by:
# 
# \begin{equation}
# r(\theta) = \left( \left| \frac{\cos\left(\frac{m \theta}{4}\right)}{a} \right|^{n_2} + \left| \frac{\sin\left(\frac{m \theta}{4}\right)}{b} \right|^{n_3} \right)^{-\frac{1}{n_1}}
# \end{equation}
# 
# where:
#  $r(\theta)$ defines the radial distance as a function of the polar angle $\theta$.
#  $m$ determines the number of lobes (periodic symmetry).
#  $n_1$ adjusts the overall roundness or spikiness of the shape (smaller values yield more angular shapes).
#  $n_2$ and $n_3$ influence the shape's curvature and the sharpness or smoothness of the lobes.
#  $a$ and $b$ control the scaling along the $x$ and $y$ axes (fixed to 1).
#
# In the SDL, the parameters \($n_1$, $n_2$, $n_3$\) can be randomly perturbed in each one of the
# $m$ lobes to generate unique and varied shapes that resemble biological forms.
#
# The ultimate objective of the SDL pattern is to test the performance of non-linear image processing 
# algorithms based on machine-learning techniques. High-resolution, noise-free realizations of the 
# SDL can be computationally  degraded to reproduce images acquired with a real imaging device. 
# These images can then be post-processed with denoising and super-resolution algorithms to try to 
# recover the original ground truth image. Methods developed to process the Dead Leaves phantom [5] 
# can be used for a full-reference analysis of the information recovered in the post-processing.
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
# **Date**: 2024/12/10
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

#    !pip install numpy matplotlib scipy scikit-image

import numpy as np
from time import time
import random
#import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.stats import pareto

#GIF_frames = []  # !!GIF!! Comments marked with the tag GIF can be used to create a GIF animation of the pattern generation process.


class SuperDeadLeaves:
    """
    Class to generate stochastic Super Dead Leaves charts based on overlapping random shapes defined by the superformula [Johan Gielis, 2003]. 
    The center and color of the shapes are sampled from uniform distributions, while the size follows a power-law distribution.
    Superformula shapes are divided in separate lobes, defined by the parameter m (called polygonVertices in the class inputs).
    The implemented code allows the randomization of the superformula parameters independently in each lobe. 
    This creates complex, unpredictable shapes that resemble natural objects like leaves or starfish.
    If randomization is disabled, approximately regular polygons are generated with up to 8 sides, and star shapes above 8.

    Classic Dead Leaves phantoms with circles can be generated with the option: polygon_range=[2,2], randomized=False. 
    With appropriate input values, the generated SDL charts can comply with the specifications of ISO-19567-2.
    """
            
    def __init__(self, image_size=[512,512], seed=None, polygon_range=[2, 9], randomized=True, contrast=1.0, background_color=0.5, num_samples_chart=10000, rmin=0.005, rmax=0.2, borderFree=False, PowerLawExp=3):
        """
        Initialize the SuperDeadLeaves class and its superformula parameters.
        
        Args:
            image_size (int, int): The dimensions of the image in x and y. Default: [512,512] pixels. A single integer can be input for a square image.
            contrast (float): Controls the contrast of the shapes (0 to 1).
            seed (int): random nunmber generator initialization seed (ie, image ID).
            polygonVertices (int, int): minimum and maximum number of vertices in the sampled polygon. Eg: [3,3]=triangles, [3,4]=triangles and squares. 
            randomized (bool): randomize the parameters of the superformula in each lobe to get a variety of shapes (if False, use a single set of random parameters for the entire shape)
            num_samples_chart (int): Number of shapes to be sampled. Should be large enough to cover the entire image.
            rmin (float): Minimum shape radius (0 to 1).
            rmax (float): Maximum shape radius (0 to 1).
            borderFree (bool): discard shapes touching the border of the image: the default background color will surround the generated patterns.
            PowerLawExp (float): Exponent of the (inverse) power law probability distribution used to sample the radii of the shapes. Use 3 for scale-invariant patterns: prob. ~ 1/f^3. In practice, the random values are sampled using the related Pareto distribution (scipy.stats.pareto) defined by prob ~ b/f^(b+1).
        """
        # ** Chart parameters:
        self.rng = np.random.default_rng(seed)  # Initialize a local random number generator with the input seed to be able to replicate the chart
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        self.image_size = image_size
        self.polygon_range = polygon_range
        self.contrast = np.clip(np.fabs(contrast), 0.0, 1.0)
        self.randomized = randomized
        self.background_color = background_color
        self.num_samples_chart = num_samples_chart
        self.rmin = rmin
        self.rmax = rmax
        self.borderFree = borderFree
        self.PowerLawExp = PowerLawExp

        # ** Default superformula parameters (will be later changed if randomized is True):
        # NOTE: the default values generate interesting random shapes, but were not optimized in any objective way. 

        self.m = (polygon_range[0]+polygon_range[1])//2
        self.n1, self.n2, self.n3 = 3.0, 8.0, 4.0
        self.a , self.b = 1.0, 1.0
               
        # Variability in the random sampling of the parameters in each lobe (0.5 -> randomly modify variable +- 50%)
        self.variability_n1 = 0.5  
        self.variability_n2 = 0.75
        self.variability_n3 = 0.75       

        # Auxiliar internal parameters for the pattern generation:
        self.ReportingInterval = 10000   # Report pattern generation stats every time this amount of shapes are generated
        self.fractionUncovered = 0.0005  # Early termination of the pattern generation (suring a reporting event) when fewer than this fraction of pixels remain uncovered (eg, 0.0005 = 0.05% of pixels uncovered). 

        self.num_points_polygon = np.max(image_size) + 100  # Resolution of the shape sampling in polar coordinates
        self.regularPolygon = False   # If True, and randomization=False, use regular polygons as the shapes.
 
        

        
    def superformula(self, m: int, a: float, b: float, n1: float, n2: float, n3: float, theta: list):
        """
        Compute the radius r(θ) for the given list of angles θ using Johan Gielis' superformula with the input parameters.
        Reference: J. Gielis. "A generic geometric transformation that unifies a wide range of natural
                   and abstract shapes." American journal of botany 90, p. 333-338 (2003)
        """
        return (np.abs(np.cos(m*theta/4.0)/a)**n2 + np.abs(np.sin(m*theta/4.0)/b)**n3)**(-1.0/n1)

    
        
    def set_regular_polygon(self):
        """
        Define superformula parameters that look approximately like regular polygons from Gielis paper.
        Regular polygons (with up to 8 sides) are generated when randomization is disabled.
        """
        if self.m < 3:
            self.m = 2     # Circle
            self.n1, self.n2, self.n3 = 2.0, 2.0, 2.0
        elif self.m == 3:   # Triangle
            self.n1, self.n2, self.n3 = 1000.0, 1980.0, 1980.0
        elif self.m == 4:   # Square
            self.n1, self.n2, self.n3 = 1.0, 1.0, 1.0
        elif self.m == 5:   # Pentagon
            self.n1, self.n2, self.n3 = 1000.0, 620.0, 620.0
        elif self.m == 6:   # Hexagon
            self.n1, self.n2, self.n3 = 1000.0, 390.0, 390.0
        elif self.m == 7:   # Heptagon
            self.n1, self.n2, self.n3 = 1000.0, 320.0, 320.0
        elif self.m == 8:   # Octagon
            self.n1, self.n2, self.n3 = 1000.0, 250.0, 250.0
        else:               # Starfish shape
            self.n1, self.n2, self.n3 = 9.0, 25.0, 25.0
        return self.n1, self.n2, self.n3
    


    
    def generate_polygon(self):
        """
        Generates a shape based on the Superformula, randomly perturbing the n1,n2,n3 parameters in each lobe if requested.
        Returns:
            Two lists with the X, Y coordinates of the polygon vertices
        """        
        # Generate polar coordinates theta values from 0 to 2*pi
        theta = np.linspace(0.0, 2.0*np.pi, self.num_points_polygon)
        r = np.ones_like(theta)   # Init radii array with the default values for circles (radius=1 at every angle)

#        if self.randomized==False:
#            nn1, nn2, nn3 = self.set_regular_polygon()

        if self.m>0:  
            # Using superformula shapes, not circles. Iterate for each shape lobe:
            for i in range(self.m):
                start_angle = i * (2.0*np.pi/self.m)
                end_angle = (i + 1) * (2.0*np.pi/self.m)
                angle_range = (theta >= start_angle-1e-7) & (theta <= end_angle+1e-7)

                if self.randomized or i==0:
                    if self.regularPolygon:
                        # Set parameters that look like regular polygon shapes
                        nn1, nn2, nn3 = self.set_regular_polygon()   
                    else:
                        # Sample a set of superformula parameters using a uniform distribution between the default values and the values times the variability factors.
                        nn1 = self.n1 + self.n1*(self.rng.uniform(-self.variability_n1, self.variability_n1))
                        nn2 = self.n2 + self.n2*(self.rng.uniform(-self.variability_n2, self.variability_n2))
                        nn3 = self.n3 + self.n3*(self.rng.uniform(-self.variability_n3, self.variability_n3))
                        if np.fabs(nn1)<0.1:
                            nn1 = 0.1  # Avoid raising to large powers that give infinite r
                            
                # Calculate radius for the current list of lobe segment angles:
                r[angle_range] = self.superformula(self.m, self.a, self.b, nn1, nn2, nn3, theta[angle_range])
                   
        # Convert polar coordinates (r, theta) to Cartesian coordinates (x, y), with a random angular shift to randomize the shape orientation:
        theta0 = self.rng.uniform(0.0, 2.0*np.pi)
        X = r * np.cos(theta+theta0)
        Y = r * np.sin(theta+theta0)
        
        return X, Y 


    
    def generate_mask_from_polygon(self, X_vertex: float, Y_vertex: float, x_center: float=0.5, y_center: float=0.5, area_scale: float=0.1):
        """
        Generates a bitmap mask of the polygon stored in the class instance, previously generated by the "generate_polygon" function.
        All pixels inside the polygon are set to 255 (white) and all outside are set to 0 (black).
        The key feature of this function is the use of the pillow (PIL) library "polygon" function in a "Draw" object to fill the interior of the input polygonal shape.
        Args:
            x (float array): x coordinate of the polygon vertices.
            y (float array): y coordinate of the polygon vertices.
            x_center (float array): sampled center of the shape in the image, x coordinate.
            y_center (float array): sampled center of the shape in the image, y coordinate.
            area_scale (float): scaling factor of the polygon area
        Returns:
            numpy.ndarray (float): 2D numpy array with the polygon outline filled with color.
    """
        # Create a blank image
        mask = Image.new('L', self.image_size[::-1], color=0)    # !!DeBuG!! Do I need to reverse the image to be compatible with numpy for rectangular images??
    
        # Create a draw object
        draw = ImageDraw.Draw(mask)

        if self.m>0:
            # Calculate the area of the polygon using the Shoelace formula (Trapezoid equation).
            area_polygon = 0.5 * abs(np.dot(X_vertex, np.roll(Y_vertex, -1)) - np.dot(Y_vertex, np.roll(X_vertex, -1)))  #  numpy-optimized method
            # area_polygon = 0.5 * abs(sum(X_vertex[i] * Y_vertex[i + 1] - Y_vertex[i] * X_vertex[i + 1] for i in range(-1, len(X_vertex) - 1)))  # Non-optimized version. Very slow for large polygons.
        else:
            area_polygon = np.pi   # Circle with radius=1

        # Scale the polygon units to the have the same area as if we had sampled a circle, move the origin to the sampled center, and transform to pixel coordinates by truncation:
        scaling_factor = np.sqrt(area_scale/area_polygon)
        nx = ((X_vertex*scaling_factor + x_center) * self.image_size[0]).astype(int)
        ny = ((Y_vertex*scaling_factor + y_center) * self.image_size[0]).astype(int)  # using image_size[0] not [1] for rectangular images with aspect ratio != 1
        
        # Create a list of points for the polygon and fill with with white color (255)
        points = list(zip(nx, ny))
        draw.polygon(points, fill=255)    
        #     image.save(fileName)
    
        return np.array(mask)  # Return the mask as a numpy array
        


    def generate_chart(self, filename: str=None):
        """
        Generate a Super Dead Leaves chart with randomized superformula shapes. 
        The chart generation parameters are based on ISO-19567-2.
        The shape radii are sampled from a power-law distribtuion using scipy.stats.pareto.rvs()
        Args:
            filename (str): Optional parameter to save the chart in TIFF format.
        Returns:
            numpy.ndarray (float32): The generated Super Dead Leaves image.
        """    
        time_00 = time()

        # Initialize the image canvas with 'NaN' indicating uncovered areas
        img = np.full(self.image_size, np.nan).astype(np.float32)
        AspectRatio = self.image_size[1]/self.image_size[0]   # enable rectangular images
        rmin_tmp = self.rmin  # rmin will be increased near the end to cover the tiny holes that are hard to fill with small shapes in high res images        

        # Shape sampling and stacking loop:
        ii=0
        ii_last = 0
        while(ii < self.num_samples_chart):
            # Output information on the phantom generation at regular intervals:
            if (ii+1)%self.ReportingInterval==0 and ii!=ii_last:
                ii_last = ii  # Do not report again if last shape is being resampled
                # Check uncovered pixels in the central part of the image (80% of X and Y):
                central_region_NaN = np.count_nonzero(np.isnan(img[int(0.1*img.shape[0]):int(0.9*img.shape[0]), int(0.1*img.shape[1]):int(0.9*img.shape[1])]))
                central_region_NaN_fraction = central_region_NaN/(0.8*img.shape[0]*0.8*img.shape[1])
                time_ii = time() - time_00
                print(f' - ... {ii+1} of {self.num_samples_chart} shapes ({100*(ii+1)/self.num_samples_chart:4.1f}%):\t uncovered pixels = {100*central_region_NaN_fraction:8.5f}% ;\t Time: {time_ii:.5} s ;\t ETA: {(time_ii*self.num_samples_chart)/(ii+1)-time_ii:.5} s = {((time_ii*self.num_samples_chart)/(ii+1))/60-time_ii/60:.4} min ...')   #  80% central region

                # Increase rmin x10 when less than 0.5% of pixels remain uncovered, to speedup covering the last small holes:
                if central_region_NaN_fraction < 0.005 and rmin_tmp<1.1*self.rmin:
                    rmin_tmp = np.clip(self.rmin*10, 0, self.rmax*0.5)
                    # print(f"   Increasing rmin x10 to {rmin_tmp} to speedup covering the remaining small holes.")

                # Stopping the shape sampling when less fractionUncovered (eg, 0.0002 = 0.02%) of the pixels remain uncovered. The final holes will be assigned the input background value:
                if central_region_NaN_fraction < self.fractionUncovered:
                    ii = ii+1
                    break

            # Sample the radius from power-law dist, ie, a Pareto dist with power exponent - 1. Scaling to the requested rmin, and rejecting values above rmax
            r = np.inf
            while r>self.rmax:
                r = pareto.rvs(b=(self.PowerLawExp-1), scale=rmin_tmp, random_state=self.rng)
            
            #radii.append(r)   # Optionally, store the sampled radii in a list for validating the prob distribution
    
            # Randomly sample a central position:
            if self.borderFree is False:
                # Sample center including a margin (ISO-19567-2) outside the chart equal to half the max radius
                # (should be full radius but large radii almost never appear and we don't want tot reject most samplings).
                # Reject sample if the center is farther from the edge than the sampled radius (so the figure will never be visible in the chart):
                x = ((1 + self.rmax)*self.rng.random() - self.rmax/2) * AspectRatio
                y =  (1 + self.rmax)*self.rng.random() - self.rmax/2
                if (x+r) < 0.0 or (x-r) > AspectRatio or (y+r) < 0.0 or (y-r) > 1.0:
                    continue
            else:
                # Sample shape center inside the chart, and reject it if it might cross the edge of the chart:
                x = self.rng.random() * AspectRatio
                y = self.rng.random()
                if (x-r) < 0.0 or (x+r) > AspectRatio or (y-r) < 0.0 or (y+r) > 1.0:
                    continue


            # Randomly sample a color (albedo):
            c = 0.5 + (self.rng.random()-0.5)*self.contrast
                
            # Generate a new superformula-based shape with a random number of lobes:
            self.m = self.rng.integers(low=self.polygon_range[0], high=self.polygon_range[1]+1)
            X_vertex, Y_vertex = self.generate_polygon()   # Generate polygon outline
            
            shape_area = np.pi*r**2   # The random shapes will be scaled to match the area of the equivalent circle of radius r in the DL model

            shape_mask = self.generate_mask_from_polygon(X_vertex, Y_vertex, x, y, shape_area)  # Generate a 2D bitmap with 0 outside the shape, 255 inside
    
            # Create a mask for the shape replacing pixels not yet covered (new shapes added "below" old ones, not covering them)
            mask = np.isnan(img) & (shape_mask!=0)
                
            if self.borderFree:
                # Reject any shape that touches the outer edge of the image (ie, have a value in first/last row/column)
                if np.any(mask[0,:]) or np.any(mask[-1,:]) or np.any(mask[:,0]) or np.any(mask[:,-1]):
                    continue

            # Count the newly sampled shape
            ii = ii+1

            # Check if any uncovered pixel is covered by the new shape
            if np.any(mask):
                # Update the canvas with the new shape
                img[mask] = c

                # Break the loop if the image has been filled with shapes
                if not np.any(np.isnan(img)):
                    break

            #if (ii<250 and (ii&(ii-1))==0) or ii%250==0:     # !!GIF!! Iteration is a power of 2 or multiple of 500            
            #    print(f'...GIF frame {ii}...')
            #    GIF_frames.append((plt.cm.RdYlGn(img)*255).astype(np.uint8))  # Normalize and apply colormap   # !!GIF!!
            
        

        # Replace any remaining 'NaN' values with background color:
        hole_mask = np.isnan(img)
        img[hole_mask] = self.background_color
        print(f" - Final amount of shapes stacked on the chart: {ii}. Uncovered pixels set to background ({self.background_color}): {np.count_nonzero(hole_mask)} ({100*np.count_nonzero(hole_mask)/(img.shape[0]*img.shape[1]):6.3}%)")


        print(f"\n     [generate_chart]: img.shape={img.shape} , mean(img)={np.mean(img):.3} , max(img)={np.max(img):.3} , min(img)={np.min(img):.3} , img[{img.shape[0]//2},{img.shape[1]//2}]={img[img.shape[0]//2,img.shape[1]//2]:.3} , type={type(np.ravel(img)[0])}\n")   # !!DeBuG!!

        # Make sure all values are in the [0,1] interval, as it should be:        
        if np.min(img)<0.0 or np.max(img)>1.0:
            print(f"     WARNING: found pixel with values smaller than 0 or larger than 1!? Clipping them to the [0,1] interval.\n")
            img = np.clip(img, 0.0, 1.0)

        #GIF_frames.append((plt.cm.RdYlGn(img)*255).astype(np.uint8))   # !!GIF!! Save final image
        
        # Optionally, save the generated chart to an external file (if TIFF format, use loss-less compression):
        if isinstance(filename, str):
            print(f" - Saving the chart image in file \"{filename}\".")
            image = Image.fromarray(img)
            image.save(filename, compression="tiff_deflate")
        return img    

        


    def generate_cell_chart(self, cell_spacing_area_factor: float=1.0, cell_nucleus_area_factor: float=0.25, cell_nucleus_intensity: float=1.25, filename: str=None):   # !!CELL!!
        """
         Simple extension to the SDL to generate a chart with superformula shapes that resemble isolated cells on a microscope slide (eg, a liquid-based pap test).
    
        Returns:
        - numpy.ndarray (float32): The generated Super Dead Leaves image with isolated shapes with a central nucleus.
        """
        # Initialize the image canvas with 'NaN' indicating uncovered areas
        img = np.full(self.image_size, np.nan).astype(np.float32)        
        cell_boundary_mask = np.full(self.image_size, False).astype(bool)   # !!CELL!! Mask that is True for all pixels covered by a cell or in the desired empty space between cells.    
        AspectRatio = self.image_size[1]/self.image_size[0]   # Enable generating rectangular phantoms: 
    
        ii = 0
        iteration = -1
        iteration_MAX = 10*self.num_samples_chart  # !!CELL!! Try sampling at most this amount of shapes before giving up (many shapes will be rejected due to overlapping).
        
        while ii<self.num_samples_chart and iteration<iteration_MAX:
            iteration = iteration + 1            
            if iteration>0 and iteration%500==0:
                print(f'... {ii} shapes of {self.num_samples_chart} ({100.0*ii/self.num_samples_chart}%) added to the chart (loop iteration {iteration})...')
            
            # Sample the radius from power-law dist, ie, a Pareto dist with power exponent - 1. Scaling to the requested rmin, and rejecting values above rmax
            r = np.inf
            while r>self.rmax:
                r = pareto.rvs(b=(3-1), scale=self.rmin, random_state=self.rng)  

            shape_area = np.pi*r**2   # The random shapes will be scaled to match the area of the equivalent circle of radius r in the DL model
                
            # Randomly choose a central position and color (albedo)
            x = self.rng.random() * AspectRatio
            y = self.rng.random()
            c = 0.5 + (self.rng.random()-0.5)*self.contrast
             
            # Generate a new superformula-based shape with a random number of lobes:
            self.m = self.rng.integers(low=self.polygon_range[0], high=self.polygon_range[1]+1)
            X_vertex, Y_vertex = self.generate_polygon()   # Generate polygon outline
            expanded_shape_mask = self.generate_mask_from_polygon(X_vertex, Y_vertex, x, y, shape_area*cell_spacing_area_factor)  # !!CELL!! Generate a 2D bitmap mask for the cell and empty space around it
    
            if self.borderFree:
                # Reject any new shape that touches the outer edges of the image (ie, any true value in first/last row/column)
                if np.any(expanded_shape_mask[0,:]) or np.any(expanded_shape_mask[-1,:]) or np.any(expanded_shape_mask[:,0]) or np.any(expanded_shape_mask[:,-1]):
                    continue
                    
            if np.any(cell_boundary_mask & (expanded_shape_mask!=0)):
                continue   # !!CELL!! Reject the shape if any pixel in the expanded mask is already True in the cell boundary mask
            
            cell_boundary_mask = cell_boundary_mask | (expanded_shape_mask!=0)   # !!CELL!! Add new mask

            # Update the canvas with the new shape, and a smaller version of the shape in the center as a 1st approximation nucleus model:
            shape_mask          = self.generate_mask_from_polygon(X_vertex, Y_vertex, x, y, shape_area)
            reduced_shape_mask  = self.generate_mask_from_polygon(X_vertex, Y_vertex, x, y, shape_area*cell_nucleus_area_factor)  # !!CELL!! Mask for the interior nucleus

            img[shape_mask!=0] = c            
            img[reduced_shape_mask!=0] = c * cell_nucleus_intensity
            
            #if (ii<25 and (ii&(ii-1))==0) or ii%25==0:     # !!GIF!! Iteration is a power of 2 or multiple of 500            
            #    print(f'...GIF frame {ii}...')
            #    GIF_frames.append((plt.cm.RdPu(img)*255).astype(np.uint8))  # Normalize and apply colormap   # !!GIF!!
            
            ii = ii + 1  # Count that a new shape was added to the chart

                
        # Replace any remaining 'NaN' values with background color:
        img[np.isnan(img)] = self.background_color

        #GIF_frames.append((plt.cm.RdPu(img)*255).astype(np.uint8))   # !!GIF!! Save final image
        
        # Optionally, save the generated chart to an external file (if TIFF format, use loss-less compression):
        if isinstance(filename, str):
            print(f" - Saving the chart image in file \"{filename}\".")
            image = Image.fromarray(img)
            image.save(filename, compression="tiff_deflate")
        
        print(f"\n  - Number of cells included in the chart: {ii}\n")        
        return img
    


    def report(self):
        """
        Report the value and type of the internal class variables.
        """
        print("\n ** [REPORT] SuperDeadLeaves class instance variables:")
        attributes = vars(self)
        for attr_name, value in attributes.items():
            print(f"       - {attr_name} = {value},\t type={type(value)}")
        print("\n")



#######################################################################################


# Auxiliary functions:

def save_image_batch_tiff(image_list, filename):
    """
    Save the list of images to a single multi-page TIFF image.
    """
    pil_images = [Image.fromarray(img) for img in image_list]  # Convert each NumPy array in the list to a PIL Image    
    pil_images[0].save(
        filename, 
        save_all      = True,            # Enable saving multiple frames in a single file
        append_images = pil_images[1:],  # Append the rest of the images
        compression   = "tiff_deflate"   # Optional: compress the TIFF
    )


#######################################################################################


# Example generation of a Super Dead Leaves chart using superformula shapes:
# (Run python with -u option to disable output buffering and see output comments immediately.)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    time0 = time()
    image_size = [1024, 1024]  # [1080, 1920]
    seed = np.random.randint(1e4, 1e5)
    rmin = 0.0025
    rmax = 0.2
    num_samples   = 100000

    # - Create a Super Dead Leaves class instance:
    SDL = SuperDeadLeaves(image_size=image_size, seed=seed, polygon_range=[3,10], randomized=True, contrast=1.0, rmin=rmin, rmax=rmax, num_samples_chart=num_samples) 
    
    # - Generate a sample SDL chart:
    print(f" ** Generating a \"Super Dead Leaves\" pattern chart:")
    print(f"    - Parameters: image_size = {image_size} pixels, seed={seed}, contrast={SDL.contrast}, num_samples={SDL.num_samples_chart}, rmin={SDL.rmin}, rmax={SDL.rmax}")
    print(f"                  polygon_range={SDL.polygon_range}, randomized={SDL.randomized}, borderFree={SDL.borderFree}")

    chart = SDL.generate_chart(f'SDL_{seed}_{SDL.polygon_range[0]}-{SDL.polygon_range[1]}_rmin{SDL.rmin}_rmax{SDL.rmax}_{image_size[0]}x{image_size[1]}.tif')    
    SDL.report()  # Report the chart parameters in the class instance internal variables

    plt.figure(figsize=(10, 10))
    plt.imshow(chart, cmap='RdYlGn', vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
  
    print(f"\n - Time used to generate the SDL image: {(time()-time0):.5} s = {(time()-time0)/60:.4} min\n\n")


    # - Create a traditional Dead Leaves chart with circles:
#    time0 = time()
#    seed = np.random.randint(1e4, 1e5)
#    print(f" ** Generating a traditional \"Dead Leaves\" pattern chart (seed={seed}):")

#    DL = SuperDeadLeaves(image_size=image_size, seed=seed, polygon_range=[2,2], randomized=False, rmin=subpixel_rmin, num_samples_chart=num_samples)   # Generate only circles
#    chart_DL = DL.generate_chart(f'DL_{seed}_{DL.polygon_range[0]}-{DL.polygon_range[1]}_rmin{DL.rmin}_rmax{DL.rmax}.tif')
#    DL.report()  # Report the chart parameters in the class instance internal variables
    
    #plt.figure(figsize=(10, 10))  #figsize=(4, 4), dpi=600)
    #plt.imshow(chart_DL, cmap='RdYlGn', vmin=0, vmax=1)     #  my_cm    , cmap='gray', vmin=0, vmax=1)  'YlGn' 'RdYlGn'
    #plt.axis('off')
    #plt.tight_layout()
    #plt.show()
    
#    print(f"\n - Time used to generate DL the image: {(time()-time0):.5} s = {(time()-time0)/60:.4} min\n\n")


    # Extension of SDL to generate a random cell chart:
#    time0 = time()
#    seed = np.random.randint(1e4, 1e5)
#    cSDL = SuperDeadLeaves(image_size=image_size, seed=seed, polygon_range=[2,6], randomized=True, borderFree=True)
#    cSDL.num_samples_chart = 200
#    SDL.contrast = 0.5
#    cSDL.rmin = 0.015
#    cSDL.rmax = 0.300
#    cSDL.variability_n1 = 0.2
#    cSDL.background_color = 0.0

#    print(f" ** Generating a cell pattern chart based on the \"Super Dead Leaves\" algorithm:")
#    print(f"    - Parameters: image_size = {image_size} pixels, seed={seed}, contrast={cSDL.contrast}, num_samples={cSDL.num_samples_chart}, rmin={cSDL.rmin}, rmax={cSDL.rmax}")
#    print(f"                  polygon_range={SDL.polygon_range}, randomized={cSDL.randomized}, borderFree={cSDL.borderFree}")

#    fname = f'cell_SDL_{seed}.tif'  # None
#    cell_chart = cSDL.generate_cell_chart(cell_spacing_area_factor=4.0, cell_nucleus_area_factor=0.10, cell_nucleus_intensity=1.5, filename=fname)  # !!CELL!!)  # (f'cellSDL_{seed}.tif')
#    cSDL.report()  # Report the chart parameters in the class instance internal variables

#    print(f"\n - Time used to generate the cell-DL image: {(time()-time0):.3} s\n\n")

    #plt.figure(figsize=(10, 10))  #figsize=(4, 4), dpi=600)
    #plt.imshow(cell_chart, cmap='RdPu', vmin=0, vmax=1)     #  RdPu    , cmap='gray', vmin=0, vmax=1)  'YlGn' 'RdYlGn'
    #plt.axis('off')
    #plt.tight_layout()
    #plt.show()      
    

    # Create a GIF animation with chart generation frames:
    # Uncomment all the lines above with the comment !!GIF!!
    #import imageio
    #fps = 3
    #imageio.mimsave(f'cell_SDL_{seed}_animation.gif', GIF_frames, fps=fps)
    #print(f'Creating GIF animation with {len(GIF_frames)} frames at {fps} fps.')
