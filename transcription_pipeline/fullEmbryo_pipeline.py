import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from skimage import color, filters, morphology, util, measure, transform#, exposure, segmentation, io
from scipy.spatial import ConvexHull
from skimage.draw import line
import feret
import cv2 as cv
from skimage.transform import AffineTransform

class FullEmbryo:
    def __init__(self, full_embryo, img_window, his_channel):
        # Full embryo dataset and metadata
        self.full_embryo_dataset_surf = full_embryo.channels_full_dataset_surf
        self.full_embryo_dataset_mid = full_embryo.channels_full_dataset_mid
        self.full_embryo_metadata_surf = full_embryo.export_global_metadata_surf
        self.full_embryo_metadata_mid = full_embryo.export_global_metadata_mid
        # ImgWindow dataset and metadata
        self.img_window_dataset = img_window.channels_full_dataset
        self.img_window_metadata = img_window.export_global_metadata
        self.his_channel = his_channel

        # Uninitialized variables
        self.ap_line = None
        self.ap90_line = None
        self.ap_d = None
        self.ap90_d = None
        self.ap90_points = None
        self.anterior_point = None
        self.posterior_point = None


        self.max_loc = None
        self.ap_angle = None
        self.im_shape_scaled = None

        self.full_embryo_mask = None
        self.conv_result = None

    @staticmethod
    def contour_mask(binary_mask):
        """
        Generates a mask by flood filling the largest contour within the input binary_mask.
        """
        contours = measure.find_contours(binary_mask)

        # Identify the desired contour (e.g., the largest)
        largest_contour = max(contours, key=len)

        # Fit a convex hull to the contour
        hull = ConvexHull(largest_contour)

        # Initialize the FullEmbryo mask
        mask = np.zeros(binary_mask.shape)

        # Extract points from the contour
        pts0 = [(largest_contour[simplex, 1][0], largest_contour[simplex, 0][0]) for simplex in hull.simplices]
        pts1 = [(largest_contour[simplex, 1][1], largest_contour[simplex, 0][1]) for simplex in hull.simplices]
        pts = pts0 + pts1
        pts = np.array(pts)

        # Calculate reference point for determining polar angle
        reference_point = np.mean(pts, axis=0)

        # Function to calculate the polar angle relative to a reference point
        def polar_angle(point):
            x, y = point[0] - reference_point[0], point[1] - reference_point[1]
            return np.arctan2(y, x)

        # Sort points based on polar angle
        sorted_pts = sorted(pts, key=polar_angle)

        # Draw contour connecting sorted points
        for i in range(len(sorted_pts)):
            if i == len(sorted_pts) - 1:
                x1, y1 = np.round(sorted_pts[i])
                x2, y2 = np.round(sorted_pts[0])
            else:
                x1, y1 = np.round(sorted_pts[i])
                x2, y2 = np.round(sorted_pts[i + 1])

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            rr, cc = line(y1, x1, y2, x2)
            mask[rr, cc] = 1

        # Save contour mask
        contour_mask = mask

        # Flood fill to generate the FullEmbryo mask
        mask = morphology.flood_fill(mask, (0, 0), 1, connectivity=1)
        mask = util.invert(mask)

        return mask, contour_mask

    def gen_full_embryo_mask(self, tif_array, sigma=10, radius=5):
        """
        Creates a FullEmbryo mask by detecting the embryo edge through a Gaussian blur, thresholding, and a closing operation.
        """
        # Convert the image to grayscale if it's not already
        # if tif_array.shape[-1] == 3:
        #     grayscale_image = color.rgb2gray(tif_array)
        # else:
        #     grayscale_image = tif_array

        # Gaussian blur the image with given sigma
        tif_array = filters.gaussian(tif_array, sigma)

        # Otsu thresholding
        threshold_value = filters.threshold_otsu(tif_array)
        tif_array = tif_array > 1 * threshold_value

        # Closing with disk of given radius
        tif_array = morphology.closing(tif_array, morphology.disk(radius))

        mask, contour = self.contour_mask(tif_array)
        return mask, contour

    @staticmethod
    def find_cross_points(img, contour, coord1, angle, t):
        """
        Draws the lines which run through the maxferet and minferet.
        """
        ymax, xmax = img.shape
        # xs = np.linspace(0, xmax, 2)
        if angle == np.pi / 2:
            x1 = coord1[1]
            y1 = 0
            x2 = coord1[1]
            y2 = ymax
        elif angle == 0:
            x1 = 0
            y1 = coord1[0]
            x2 = xmax
            y2 = coord1[0]
        else:
            x1 = 0
            y1 = t
            x2 = xmax
            y2 = np.tan(angle) * xmax + t

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        mask_height, mask_width = img.shape

        def is_inside_mask(x, y):
            return 0 <= x < mask_width and 0 <= y < mask_height

        def is_intersection(x, y):
            if contour[y, x] == 1:
                return True
            try:
                if contour[y - 1, x] == 1 or contour[y + 1, x] == 1:
                    return True
                elif contour[y, x - 1] == 1 or contour[y, x + 1] == 1:
                    return True
                else:
                    return False
            except:
                return False

        def is_point_inside_mask(x, y):
            return is_inside_mask(x, y) and is_intersection(x, y)

        intersections = []

        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if is_point_inside_mask(x1, y1):
                intersections.append((x1, y1))

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        # Filter duplicate intersection points
        intersections = list(set(intersections))

        intersections = np.array(intersections)

        # Collapse points within a certain radius to the average point to reduce the number of detected intersections
        def reduce_points(points, max_distance):
            reduced_points = []

            while len(points) > 0:
                current_point = points[0]
                points = points[1:]

                close_points = [current_point]

                for p in points:
                    distance = np.linalg.norm(current_point - p)
                    if distance <= max_distance:
                        close_points.append(p)

                average_point = np.mean(close_points, axis=0)
                reduced_points.append(average_point)

                points = [p for p in points if np.linalg.norm(average_point - p) > max_distance]

            return np.array(reduced_points)

        unique_intersections = reduce_points(intersections, 5)

        return unique_intersections

    def find_ap_axis(self, make_plots=False, sigma=10, radius=5, remove_small_objects=False, ap_method='minf90'):
        """
        Main method to find the Anterior-Posterior axis.
        Users can press 'm' to manually select anterior and posterior points,
        then define the AP_perp axis by clicking where it intersects the embryo boundaries.
        """
        # Calculate the Full embryo mask using the mid z-slice from the Mid image

        if remove_small_objects:
            # Remove small objects from the mid image
            # Get the image data
            mid_image = self.full_embryo_dataset_mid[self.his_channel][1, :, :]

            # Apply Otsu's thresholding method
            thresh = filters.threshold_otsu(mid_image)
            binary_mask = mid_image < thresh
            mid_image = np.where(binary_mask, mid_image, 0)
            self.full_embryo_mask, contour = self.gen_full_embryo_mask(mid_image, sigma, radius)
        else:
            # Use the original mid image without thresholding
            self.full_embryo_mask, contour = self.gen_full_embryo_mask(
                self.full_embryo_dataset_mid[self.his_channel][1, :, :],
                sigma, radius)

        # Calculate the Feret diameters
        feret_result = feret.calc(self.full_embryo_mask)

        '''
        The following code finds the cross points of the AP and AP90 lines. Depending on the method chosen, it will
        use either the minf90 or maxf method to find the cross points. The AP and AP90 lines are then drawn on the
        image. The points are stored in the class variables ap_line and ap90_line. The distance between the AP points
        and the centroid of the embryo is also calculated.
        '''
        if ap_method == 'minf90':
            ap_pts = self.find_cross_points(self.full_embryo_mask, contour, feret_result.minf90_coords[0],
                                            feret_result.minf90_angle, feret_result.minf90_t)
            ap90_pts = self.find_cross_points(self.full_embryo_mask, contour, feret_result.minf_coords[0],
                                              feret_result.minf_angle, feret_result.minf_t)
            self.ap_angle = feret_result.minf90_angle
        elif ap_method == 'maxf':
            ap_pts = self.find_cross_points(self.full_embryo_mask, contour, feret_result.maxf_coords[0],
                                            feret_result.maxf_angle, feret_result.maxf_t)
            ap90_pts = self.find_cross_points(self.full_embryo_mask, contour, feret_result.maxf90_coords[0],
                                              feret_result.maxf90_angle, feret_result.maxf90_t)
            self.ap_angle = feret_result.maxf_angle
        else:
            print("Invalid AP method. Please choose 'minf90' or 'maxf'.")
            return

        # Find the endpoints of the AP and AP90 lines

        ## AP axis points
        (y1, x1) = ap_pts[0]
        (y2, x2) = ap_pts[1]

        ## AP90 axis points
        (y3, x3) = ap90_pts[0]
        (y4, x4) = ap90_pts[1]

        x1 = int(x1); y1 = int(y1)
        x2 = int(x2); y2 = int(y2)
        x3 = int(x3); y3 = int(y3)
        x4 = int(x4); y4 = int(y4)

        ## Draw the AP and AP90 lines
        self.ap_line = line(y1, x1, y2, x2)
        self.ap90_line = line(y3, x3, y4, x4)

        # Store ap90 points in class variables
        self.ap90_points = [(y3, x3), (y4, x4)]

        # Label connected components in the binary mask
        self.full_embryo_mask = self.full_embryo_mask.astype(int)

        label_mask = measure.label(self.full_embryo_mask)

        # Calculate object properties using regionprops
        properties = measure.regionprops(self.full_embryo_mask)

        # Calculate the centroid for each labeled object
        for idx, prop in enumerate(properties):
            centroid = prop.centroid

        x0, y0 = centroid
        # Distance between ap points and centroid
        d1 = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        d2 = np.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2)

        # Length of ap and ap90 lines in pixels
        self.ap_d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        self.ap90_d = np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)

        if d1 > d2:
            self.anterior_point = (y1, x1)
            self.posterior_point = (y2, x2)
        else:
            self.anterior_point = (y2, x2)
            self.posterior_point = (y1, x1)

        # Pull last frame of histone channel from movie and histone channel of surf image
        img_window = self.img_window_dataset[self.his_channel][-1, :, :, :]
        surf = self.full_embryo_dataset_surf[self.his_channel][:, :, :]

        # Max projection of ImgWindow and Surf
        img_window_max = np.max(img_window, axis=0)
        surf_max = np.max(surf, axis=0)

        # Pixel size for Surf
        surf_dx = self.full_embryo_metadata_surf[1]['PixelsPhysicalSizeX']
        surf_dy = self.full_embryo_metadata_surf[1]['PixelsPhysicalSizeY']
        surf_dz = self.full_embryo_metadata_surf[1]['PixelsPhysicalSizeZ']

        surf_spacing = (surf_dz, surf_dy, surf_dx)

        # Pixel size for ImgWindow
        img_window_dx = self.img_window_metadata[1]['PixelsPhysicalSizeX']
        img_window_dy = self.img_window_metadata[1]['PixelsPhysicalSizeY']
        img_window_dz = self.img_window_metadata[1]['PixelsPhysicalSizeZ']

        img_window_spacing = (img_window_dz, img_window_dy, img_window_dx)

        # Rescale ImgWindow_max to match the zoom of Surf
        img_window_max_rescaled = transform.rescale(img_window_max,
                                                    img_window_spacing[2] / surf_spacing[2],
                                                    anti_aliasing=True)

        template = img_window_max_rescaled
        img = surf_max
        img = img.astype(np.float32)
        template = template.astype(np.float32)
        w, h = template.shape[::-1]
        self.im_shape_scaled = (w, h)

        # Apply template Matching
        method = cv.TM_CCOEFF_NORMED
        conv_result = cv.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(conv_result)
        self.conv_result = conv_result
        self.max_loc = max_loc

        # Added by Josh: scale factor calculation
        w_0 = img_window_max.shape[1]  # width of the original image
        h_0 = img_window_max.shape[0]  # height of the original image
        scale_factor_w = w / w_0
        scale_factor_h = h / h_0
        scale_factor = np.mean([scale_factor_w, scale_factor_h])

        if make_plots:
            # Set up the plot for possible manual selection
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            ## Surf image
            img = np.max(self.full_embryo_dataset_surf[self.his_channel], axis=0)
            rect = patches.Rectangle((self.max_loc[0], self.max_loc[1]), self.im_shape_scaled[0],
                                     self.im_shape_scaled[1], linewidth=1, edgecolor='r', facecolor='none')
            axes[0].add_patch(rect)
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title('Overlay with Surf Image')

            ## Mid image
            rect = patches.Rectangle((self.max_loc[0], self.max_loc[1]), self.im_shape_scaled[0],
                                     self.im_shape_scaled[1], linewidth=1, edgecolor='r', facecolor='none')
            axes[1].add_patch(rect)
            axes[1].imshow(self.full_embryo_dataset_mid[self.his_channel][0, :, :], cmap='gray')
            axes[1].imshow(self.full_embryo_mask, alpha=0.2)
            axes[1].plot(self.ap_line[0], self.ap_line[1])
            axes[1].plot(self.ap90_line[0], self.ap90_line[1])

            axes[1].scatter(self.anterior_point[0], self.anterior_point[1], s=20, alpha=0.8,
                            label='Anterior', color='green')
            axes[1].scatter(self.posterior_point[0], self.posterior_point[1], s=20, alpha=0.8,
                            label='Posterior', color='magenta')

            axes[1].scatter(self.ap90_points[0][0], self.ap90_points[0][1], s=20, alpha=0.8,
                            label='AP_perp', color='blue')
            axes[1].scatter(self.ap90_points[1][0], self.ap90_points[1][1], s=20, alpha=0.8,
                            color='blue')

            axes[1].set_title('Overlay with Mid Image')
            axes[1].legend()

            # Flag to track current mode
            mode = {'current': 'view'}  # Can be 'view', 'ap_select', or 'ap_perp_select'

            # Store clicks for manual selection
            manual_points = []
            ap_perp_points = []

            # Store AP_perp line information for proximity calculation
            ap_perp_line_info = {'origin': None, 'angle': None, 'line_points': None, 'length': None}

            def get_closest_point_on_line(point, origin, angle):
                """Get point on line defined by origin and angle closest to given point"""
                # Project point onto line
                dx = point[1] - origin[1]
                dy = point[0] - origin[0]
                # Dot product with unit direction vector
                t = dx * np.cos(angle) + dy * np.sin(angle)
                # Closest point
                x = origin[1] + t * np.cos(angle)
                y = origin[0] + t * np.sin(angle)
                return x, y

            def on_key(event):
                if event.key == 'm':
                    if mode['current'] == 'view':
                        mode['current'] = 'ap_select'
                        # Clear the right plot and prepare for manual selection
                        axes[1].clear()
                        axes[1].imshow(self.full_embryo_dataset_mid[self.his_channel][0, :, :], cmap='gray')
                        axes[1].imshow(self.full_embryo_mask, alpha=0.2)
                        axes[1].set_title('Click to select Anterior point, then Posterior point')
                        plt.draw()
                        print("Manual mode activated. Click to select Anterior point, then Posterior point.")

            def on_click(event):
                if event.inaxes != axes[1]:
                    return  # Ignore clicks outside the right axis

                x, y = event.xdata, event.ydata

                if mode['current'] == 'ap_select':
                    manual_points.append((int(y), int(x)))  # Store as (y, x) for consistency

                    if len(manual_points) == 1:
                        # First click - Anterior point
                        axes[1].scatter(x, y, s=20, alpha=0.8, color='green', label='Anterior (Manual)')
                        axes[1].set_title('Now click to select Posterior point')
                        plt.draw()

                    elif len(manual_points) == 2:
                        # Second click - Posterior point
                        axes[1].scatter(x, y, s=20, alpha=0.8, color='magenta', label='Posterior (Manual)')

                        # Update AP axis
                        self.anterior_point = manual_points[0]
                        self.posterior_point = manual_points[1]

                        # Calculate new AP angle
                        dy = self.posterior_point[0] - self.anterior_point[0]
                        dx = self.posterior_point[1] - self.anterior_point[1]
                        self.ap_angle = np.arctan2(dy, dx)

                        # Draw new AP line
                        self.ap_line = line(self.anterior_point[0], self.anterior_point[1],
                                            self.posterior_point[0], self.posterior_point[1])

                        # Calculate perpendicular angle for AP90
                        ap90_angle = self.ap_angle + np.pi / 2

                        # Calculate midpoint of AP line
                        mid_y = (self.anterior_point[0] + self.posterior_point[0]) / 2
                        mid_x = (self.anterior_point[1] + self.posterior_point[1]) / 2
                        mid_point = (mid_y, mid_x)

                        # Calculate length of AP line
                        self.ap_d = np.sqrt((self.anterior_point[1] - self.posterior_point[1]) ** 2 +
                                            (self.anterior_point[0] - self.posterior_point[0]) ** 2)

                        # Draw AP perpendicular line that extends across the entire embryo
                        # We'll make it much longer than needed to ensure it crosses the embryo
                        img_height, img_width = self.full_embryo_mask.shape
                        extension_length = min(img_height, img_width)

                        # Calculate endpoints of extended AP90 line
                        ap90_x1 = int(mid_x + extension_length * np.cos(ap90_angle))
                        ap90_y1 = int(mid_y + extension_length * np.sin(ap90_angle))
                        ap90_x2 = int(mid_x - extension_length * np.cos(ap90_angle))
                        ap90_y2 = int(mid_y - extension_length * np.sin(ap90_angle))

                        # Store the extended line for display
                        extended_ap90_line = line(ap90_y1, ap90_x1, ap90_y2, ap90_x2)

                        # Store information about this line for calculations
                        ap_perp_line_info['origin'] = mid_point
                        ap_perp_line_info['angle'] = ap90_angle
                        ap_perp_line_info['line_points'] = [(ap90_y1, ap90_x1), (ap90_y2, ap90_x2)]

                        # Draw the AP axis and extended AP90 line
                        axes[1].plot([p for p in self.ap_line[1]], [p for p in self.ap_line[0]], 'r-',
                                     label='AP axis (Manual)')
                        axes[1].plot([p for p in extended_ap90_line[1]], [p for p in extended_ap90_line[0]], 'b--',
                                     label='Extended AP90 line')

                        # Transition to AP perpendicular selection mode
                        mode['current'] = 'ap_perp_select'
                        axes[1].set_title('Click where AP_perp line intersects embryo boundaries (2 points)')
                        axes[1].legend()
                        plt.draw()

                        print(
                            "AP axis defined. Now click where the perpendicular line intersects the embryo boundaries.")

                elif mode['current'] == 'ap_perp_select':
                    # Handling clicks for AP perpendicular points
                    # Get the clicked point
                    click_point = (y, x)  # (y, x) format for consistency

                    # Find closest point on the AP_perp line
                    closest_x, closest_y = get_closest_point_on_line(
                        (y, x),  # Input as (y, x)
                        ap_perp_line_info['origin'],  # Origin is already in (y, x) format
                        ap_perp_line_info['angle']
                    )

                    # Store the point on the line
                    ap_perp_points.append((int(closest_y), int(closest_x)))  # Store as (y, x)

                    # Display the selected point
                    axes[1].scatter(closest_x, closest_y, s=20, alpha=0.8, color='cyan',
                                    marker='x' if len(ap_perp_points) == 1 else 'o')

                    if len(ap_perp_points) == 1:
                        axes[1].set_title('Click where AP_perp intersects the other side of the embryo')
                        plt.draw()

                    elif len(ap_perp_points) == 2:
                        # Calculate distance between the two AP_perp points
                        self.ap90_d = np.sqrt((ap_perp_points[0][1] - ap_perp_points[1][1]) ** 2 +
                                              (ap_perp_points[0][0] - ap_perp_points[1][0]) ** 2)

                        # Update AP90 information
                        self.ap90_line = line(ap_perp_points[0][0], ap_perp_points[0][1],
                                              ap_perp_points[1][0], ap_perp_points[1][1])
                        self.ap90_points = ap_perp_points

                        # Redraw everything with final measurements
                        axes[1].clear()
                        axes[1].imshow(self.full_embryo_dataset_mid[self.his_channel][0, :, :], cmap='gray')
                        axes[1].imshow(self.full_embryo_mask, alpha=0.2)

                        # Draw AP line
                        axes[1].plot([p for p in self.ap_line[1]], [p for p in self.ap_line[0]], 'r-', label='AP axis')
                        axes[1].scatter(self.anterior_point[1], self.anterior_point[0], s=20, alpha=0.8,
                                        color='green', label='Anterior')
                        axes[1].scatter(self.posterior_point[1], self.posterior_point[0], s=20, alpha=0.8,
                                        color='magenta', label='Posterior')

                        # Draw AP90 line
                        axes[1].plot([p for p in self.ap90_line[1]], [p for p in self.ap90_line[0]], 'b-',
                                     label='AP90 axis')
                        axes[1].scatter(ap_perp_points[0][1], ap_perp_points[0][0], s=20, alpha=0.8,
                                        color='blue', label='AP90 point 1')
                        axes[1].scatter(ap_perp_points[1][1], ap_perp_points[1][0], s=20, alpha=0.8,
                                        color='blue', label='AP90 point 2')

                        axes[1].set_title('Manual AP and AP90 axes defined')
                        axes[1].legend()
                        plt.draw()

                        mode['current'] = 'view'  # Return to view mode

                        print(f"Manual AP and AP90 axes defined.")
                        print(f"Anterior Point: {self.anterior_point}")
                        print(f"Posterior Point: {self.posterior_point}")
                        print(f"AP Angle: {np.degrees(self.ap_angle):.2f} degrees")
                        print(f"AP distance: {self.ap_d:.2f} pixels")
                        print(f"AP90 distance: {self.ap90_d:.2f} pixels")

            # Connect event handlers
            fig.canvas.mpl_connect('key_press_event', on_key)
            fig.canvas.mpl_connect('button_press_event', on_click)

            plt.tight_layout()
            plt.show()

        return

    def gen_plots(self, show_conv_result=False):
        if self.conv_result is None:
            print('Please run the find_ap_axis method first.')
            print('Running find_ap_axis...')
            self.find_ap_axis() # Run the find_ap_axis method if it hasn't been run yet
            return

        if show_conv_result:
            # Plot the template matching result
            plt.figure(figsize=(12, 6))
            plt.imshow(self.conv_result, cmap='gray')
            plt.scatter(self.max_loc[0], self.max_loc[1], s=10, alpha=0.8, color='red')
            plt.title('Template Matching')
            plt.show()

        # Plot the overlay of the Surf and Mid images
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        ## Surf image
        img = np.max(self.full_embryo_dataset_surf[self.his_channel], axis=0)
        rect = patches.Rectangle((self.max_loc[0], self.max_loc[1]), self.im_shape_scaled[0],
                                 self.im_shape_scaled[1], linewidth=1, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Overlay with Surf Image')

        ## Mid image
        rect = patches.Rectangle((self.max_loc[0], self.max_loc[1]), self.im_shape_scaled[0],
                                 self.im_shape_scaled[1], linewidth=1, edgecolor='r', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].imshow(self.full_embryo_dataset_mid[self.his_channel][0, :, :], cmap='gray')
        axes[1].imshow(self.full_embryo_mask, alpha=0.2)
        axes[1].plot(self.ap_line[0], self.ap_line[1])
        axes[1].plot(self.ap90_line[0], self.ap90_line[1])

        axes[1].scatter(self.anterior_point[0], self.anterior_point[1], s=20, alpha=0.8,
                        label='Anterior', color='green')
        axes[1].scatter(self.posterior_point[0], self.posterior_point[1], s=20, alpha=0.8,
                        label='Posterior', color='magenta')

        axes[1].scatter(self.ap90_points[0][0], self.ap90_points[0][1], s=20, alpha=0.8,
                        label='AP_perp', color='blue')
        axes[1].scatter(self.ap90_points[1][0], self.ap90_points[1][1], s=20, alpha=0.8,
                        color='blue')

        axes[1].set_title('Overlay with Mid Image')
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        return

    def swap_ap_points(self, make_plots=False):
        """
        Swaps the anterior and posterior points and changes ap_angle.
        """
        self.anterior_point, self.posterior_point = self.posterior_point, self.anterior_point
        self.ap_angle = np.pi - self.ap_angle

        if make_plots:
            self.gen_plots()

        return

    import matplotlib.pyplot as plt



    def define_ap_points(self):
        """
        Allows the user to manually define the anterior and posterior points by clicking on the plot
        of the full embryo mid image. The user clicks on the anterior point first, followed by the posterior point.
        """
        # Display the mid image with the current mask overlay
        matplotlib.use('Qt5Agg')
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.full_embryo_dataset_mid[self.his_channel][0, :, :], cmap='gray')
        ax.imshow(self.full_embryo_mask, alpha=0.2)
        plt.title('Click on the anterior point first, then the posterior point.')

        # Function to capture click events
        def onclick(event):
            if onclick.count == 0:
                self.anterior_point = (int(event.ydata), int(event.xdata))
                plt.scatter(event.xdata, event.ydata, color='green', s=50, label='Anterior Point')
                plt.legend()
                plt.draw()
                onclick.count += 1
            elif onclick.count == 1:
                self.posterior_point = (int(event.ydata), int(event.xdata))
                plt.scatter(event.xdata, event.ydata, color='magenta', s=50, label='Posterior Point')
                plt.legend()
                plt.draw()
                plt.close()  # Close the plot once both points are defined
                onclick.count += 1

        onclick.count = 0  # Initialize click counter

        # Connect the click event handler to the figure
        # fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show()

        # Calculate the angle of the manually defined AP axis
        dy = self.posterior_point[0] - self.anterior_point[0]
        dx = self.posterior_point[1] - self.anterior_point[1]
        self.ap_angle = np.arctan2(dy, dx)

        # Update the AP line
        self.ap_line = line(self.anterior_point[0], self.anterior_point[1],
                            self.posterior_point[0], self.posterior_point[1])

        print(f"Anterior Point: {self.anterior_point}")
        print(f"Posterior Point: {self.posterior_point}")
        print(f"AP Angle: {np.degrees(self.ap_angle):.2f} degrees")

        return

    def xy_to_ap(self, compiled_data):

        # extract the spot position arrays from compiled_data
        x = compiled_data.x
        y = compiled_data.y

        img_window = self.img_window_dataset[self.his_channel][-1, :, :, :]
        img_window_max = np.max(img_window, axis=0)

        # Added by Josh: scale factor calculation
        w, h = self.im_shape_scaled
        w_0 = img_window_max.shape[1]  # width of the original image
        h_0 = img_window_max.shape[0]  # height of the original image
        # print(f'image size: {w_0} by {h_0}')
        scale_factor_w = w / w_0
        scale_factor_h = h / h_0
        scale_factor = np.mean([scale_factor_w, scale_factor_h])

        # Convert spot positions to the coordinates in the imaging window in the full embryo image
        scaled_x = scale_factor * x
        scaled_y = scale_factor * y

        # Shift the coordinates to the full image coordinate system
        full_image_x = scaled_x + self.max_loc[0]
        full_image_y = scaled_y + self.max_loc[1]

        # Reshape the x, y coordinates and pair them up to prep for the transformation
        reshaped_xy = np.transpose(np.stack((full_image_x, full_image_y)),
                                   (1, 0))  # exchange the axes so that the first axis is by particle, not by xy
        spots_coords = [np.stack(xy_pair, axis=1) for xy_pair in reshaped_xy]  # pair the xy coordinates into 2-tuples

        # Define the affine transformation matrix
        anterior_point = self.anterior_point
        ap_angle = self.ap_angle
        ap_d = self.ap_d
        ap90_d = self.ap90_d

        transform_matrix1 = AffineTransform(translation=(-anterior_point[0], -anterior_point[1]))
        transform_matrix2 = AffineTransform(rotation=-(ap_angle % np.pi))

        # Apply the affine transformation to the larger image coordinates and calculate the transformed coordinates in terms of (AP, AP90) percentages
        transformed_spots_ap_coords = [transform_matrix2(transform_matrix1(spot_coords)) / np.array([ap_d, ap90_d]) for
                                       spot_coords in spots_coords]

        # Split the coordinates back into AP and AP90 arrays and append them to a new row in compiled_data
        transformed_ap = [transformed_spot_ap_coords[:, 0] for transformed_spot_ap_coords in
                          transformed_spots_ap_coords]
        transformed_ap90 = [transformed_spot_ap_coords[:, 1] for transformed_spot_ap_coords in
                            transformed_spots_ap_coords]

        new_compiled_data = compiled_data.copy()
        new_compiled_data['ap'] = transformed_ap
        new_compiled_data['ap90'] = transformed_ap90

        return new_compiled_data
