import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from skimage import color, filters, morphology, util, measure, transform
from scipy.spatial import ConvexHull
from skimage.draw import line
import feret
import cv2 as cv
from skimage.transform import AffineTransform
import pandas as pd
import os


class FullEmbryo:
    def __init__(self, dataset_path, full_embryo, img_window, his_channel):
        # Full embryo data
        self.full_embryo_dataset_surf = full_embryo.channels_full_dataset_surf
        self.full_embryo_dataset_mid = full_embryo.channels_full_dataset_mid
        self.full_embryo_metadata_surf = full_embryo.export_global_metadata_surf
        self.full_embryo_metadata_mid = full_embryo.export_global_metadata_mid

        # Image window data
        self.img_window_dataset = img_window.channels_full_dataset
        self.img_window_metadata = img_window.export_global_metadata
        self.his_channel = his_channel
        self.dataset_folder = dataset_path

        # Initialize attributes to None
        self._init_attributes()

    def _init_attributes(self):
        """Initialize class attributes"""
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
        """Generate mask from largest contour"""
        contours = measure.find_contours(binary_mask)
        largest_contour = max(contours, key=len)
        hull = ConvexHull(largest_contour)
        mask = np.zeros(binary_mask.shape)

        pts = FullEmbryo._get_hull_points(largest_contour, hull)
        sorted_pts = FullEmbryo._sort_points_by_angle(pts)
        contour_mask = FullEmbryo._draw_contour(sorted_pts, mask)

        # Flood fill to generate full mask
        mask = morphology.flood_fill(contour_mask, (0, 0), 1, connectivity=1)
        mask = util.invert(mask)

        return mask, contour_mask

    @staticmethod
    def _get_hull_points(contour, hull):
        """Extract points from contour hull"""
        pts0 = [(contour[s, 1][0], contour[s, 0][0]) for s in hull.simplices]
        pts1 = [(contour[s, 1][1], contour[s, 0][1]) for s in hull.simplices]
        return np.array(pts0 + pts1)

    @staticmethod
    def _sort_points_by_angle(points):
        """Sort points by polar angle from center"""
        center = np.mean(points, axis=0)
        return sorted(points, key=lambda p:
        np.arctan2(p[1] - center[1], p[0] - center[0]))

    @staticmethod
    def _draw_contour(points, mask):
        """Draw contour connecting sorted points"""
        for i in range(len(points)):
            p1 = np.round(points[i]).astype(int)
            p2 = np.round(points[0] if i == len(points) - 1
                          else points[i + 1]).astype(int)
            rr, cc = line(p1[1], p1[0], p2[1], p2[0])
            mask[rr, cc] = 1
        return mask

    def gen_full_embryo_mask(self, tif_array, sigma=10, radius=5):
        """Create embryo mask using edge detection"""
        # Preprocess image
        blurred = filters.gaussian(tif_array, sigma)
        threshold = filters.threshold_otsu(blurred)
        binary = blurred > threshold
        closed = morphology.closing(binary, morphology.disk(radius))

        return self.contour_mask(closed)

    @staticmethod
    def find_cross_points(img, contour, coord1, angle, t):
        """Find intersection points of line with contour"""
        ymax, xmax = img.shape
        x1, y1, x2, y2 = FullEmbryo._get_line_endpoints(coord1, angle, t, ymax, xmax)

        def is_valid_point(x, y, w, h):
            return 0 <= x < w and 0 <= y < h

        def is_intersection(x, y):
            if contour[y, x] == 1:
                return True
            try:
                neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
                return any(contour[ny, nx] == 1 for ny, nx in neighbors)
            except:
                return False

        intersections = FullEmbryo._find_intersections(x1, y1, x2, y2,
                                                       lambda x, y: is_valid_point(x, y, xmax,
                                                                                   ymax) and is_intersection(x, y))

        return FullEmbryo._reduce_intersection_points(intersections)

    @staticmethod
    def _get_line_endpoints(coord1, angle, t, ymax, xmax):
        """Get endpoints for line defined by angle and intercept"""
        if angle == np.pi / 2:
            return coord1[1], 0, coord1[1], ymax
        elif angle == 0:
            return 0, coord1[0], xmax, coord1[0]
        else:
            x1, y1 = 0, t
            x2, y2 = xmax, np.tan(angle) * xmax + t
            return int(x1), int(y1), int(x2), int(y2)

    @staticmethod
    def _find_intersections(x1, y1, x2, y2, is_valid):
        """Find intersections using Bresenham's algorithm"""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        intersections = []

        while True:
            if is_valid(x1, y1):
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

        return list(set(intersections))

    @staticmethod
    def _reduce_intersection_points(points, max_dist=5):
        """Collapse nearby points to average location"""
        reduced = []
        points = np.array(points)

        while len(points) > 0:
            p = points[0]
            points = points[1:]
            close = [p]

            for other in points:
                if np.linalg.norm(p - other) <= max_dist:
                    close.append(other)

            avg = np.mean(close, axis=0)
            reduced.append(avg)
            points = [p for p in points if np.linalg.norm(avg - p) > max_dist]

        return np.array(reduced)

    def find_ap_axis(self, make_plots=False, sigma=10, radius=5,
                     load_previous=True, save_results=True, ap_method='minf90'):
        """Find anterior-posterior axis and perpendicular"""
        self.full_embryo_mask, contour = self.gen_full_embryo_mask(
            self.full_embryo_dataset_mid[self.his_channel][1, :, :],
            sigma, radius)

        if load_previous:
            self._load_previous_points()
        else:
            feret_result = feret.calc(self.full_embryo_mask)
            if not self._calculate_ap_points(feret_result, contour, ap_method):
                return
            self._calculate_scaling_parameters()

        if make_plots:
            self._setup_interactive_plot()

        return self._create_ap_points_dataframe(save_results)

    def _load_previous_points(self):
        """Try to load previously saved AP points"""
        self.ap_points_file = os.path.join(self.dataset_folder, 'ap_points_df.pkl')
        try:
            with open(self.ap_points_file, 'rb') as f:
                ap_points = pickle.load(f)
                for k, v in ap_points.items():
                    setattr(self, k, v[0])
            print("Previous AP points loaded.")
            return True
        except:
            print("No previous AP points found. Calculating AP points.")
            return False

    def _calculate_ap_points(self, feret_result, contour, ap_method):
        """Calculate AP axis points and measurements"""
        # Get intersection points based on method
        if ap_method == 'minf90':
            ap_pts = self.find_cross_points(self.full_embryo_mask, contour,
                                            feret_result.minf90_coords[0],
                                            feret_result.minf90_angle,
                                            feret_result.minf90_t)
            ap90_pts = self.find_cross_points(self.full_embryo_mask, contour,
                                              feret_result.minf_coords[0],
                                              feret_result.minf_angle,
                                              feret_result.minf_t)
            self.ap_angle = feret_result.minf90_angle
        elif ap_method == 'maxf':
            ap_pts = self.find_cross_points(self.full_embryo_mask, contour,
                                            feret_result.maxf_coords[0],
                                            feret_result.maxf_angle,
                                            feret_result.maxf_t)
            ap90_pts = self.find_cross_points(self.full_embryo_mask, contour,
                                              feret_result.maxf90_coords[0],
                                              feret_result.maxf90_angle,
                                              feret_result.maxf90_t)
            self.ap_angle = feret_result.maxf_angle
        else:
            print("Invalid AP method. Please choose 'minf90' or 'maxf'.")
            return False

        print("AP angle: ", np.degrees(self.ap_angle))
        return self._set_ap_measurements(ap_pts, ap90_pts)

    def _set_ap_measurements(self, ap_pts, ap90_pts):
        """Set AP measurements from intersection points"""
        # Extract points
        (y1, x1), (y2, x2) = ap_pts
        (y3, x3), (y4, x4) = ap90_pts

        # Convert points to int
        y1 = int(y1); x1 = int(x1)
        y2 = int(y2); x2 = int(x2)
        y3 = int(y3); x3 = int(x3)
        y4 = int(y4); x4 = int(x4)

        # Find connecting lines
        self.ap_line = line(y1, x1, y2, x2)
        self.ap90_line = line(y3, x3, y4, x4)
        self.ap90_points = [(y3, x3), (y4, x4)]

        # Calculate centroid and determine anterior/posterior
        props = measure.regionprops(self.full_embryo_mask.astype(int))[0]
        x0, y0 = props.centroid
        d1 = np.hypot(x1 - x0, y1 - y0)
        d2 = np.hypot(x2 - x0, y2 - y0)

        self.anterior_point = (y1, x1) if d1 > d2 else (y2, x2)
        self.posterior_point = (y2, x2) if d1 > d2 else (y1, x1)

        # Calculate distances
        self.ap_d = np.hypot(x1 - x2, y1 - y2)
        self.ap90_d = np.hypot(x3 - x4, y3 - y4)

        return True

    def _calculate_scaling_parameters(self):
        """Calculate scaling parameters between image sets"""
        # Get maximum intensity projections
        img_window = self.img_window_dataset[self.his_channel][-1, :, :, :]
        surf = self.full_embryo_dataset_surf[self.his_channel][:, :, :]
        img_window_max = np.max(img_window, axis=0)
        surf_max = np.max(surf, axis=0)

        # Calculate scaling factors
        surf_spacing = (self.full_embryo_metadata_surf[1]['PixelsPhysicalSizeZ'],
                        self.full_embryo_metadata_surf[1]['PixelsPhysicalSizeY'],
                        self.full_embryo_metadata_surf[1]['PixelsPhysicalSizeX'])

        img_window_spacing = (self.img_window_metadata[1]['PixelsPhysicalSizeZ'],
                              self.img_window_metadata[1]['PixelsPhysicalSizeY'],
                              self.img_window_metadata[1]['PixelsPhysicalSizeX'])

        # Apply template matching
        rescale_factor = img_window_spacing[2] / surf_spacing[2]
        template = transform.rescale(img_window_max, rescale_factor, anti_aliasing=True)
        template = template.astype(np.float32)
        img = surf_max.astype(np.float32)

        self.im_shape_scaled = template.shape[::-1]
        self.conv_result = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        _, _, _, self.max_loc = cv.minMaxLoc(self.conv_result)

    def _setup_interactive_plot(self):
        """Setup interactive plotting"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        mode = {'current': 'view'}
        manual_points = []
        ap_perp_points = []
        ap_perp_line_info = {'origin': None, 'angle': None,
                             'line_points': None, 'length': None}

        def plot_current_state():
            """Update plot with current state"""
            for ax in axes:
                ax.clear()

            # Plot surf image
            surf_img = np.max(self.full_embryo_dataset_surf[self.his_channel], axis=0)
            rect = patches.Rectangle(self.max_loc,
                                     self.im_shape_scaled[0],
                                     self.im_shape_scaled[1],
                                     linewidth=1, edgecolor='r', facecolor='none')
            axes[0].add_patch(rect)
            axes[0].imshow(surf_img, cmap='gray')
            axes[0].set_title('Overlay with Surf Image')

            # Plot mid image with overlays
            rect = patches.Rectangle(self.max_loc,
                                     self.im_shape_scaled[0],
                                     self.im_shape_scaled[1],
                                     linewidth=1, edgecolor='r', facecolor='none')
            axes[1].add_patch(rect)
            axes[1].imshow(self.full_embryo_dataset_mid[self.his_channel][0, :, :],
                           cmap='gray')
            axes[1].imshow(self.full_embryo_mask, alpha=0.2)

            # Add AP lines and points if they exist
            for attr, color in [('ap_line', 'orange'), ('ap90_line', 'b')]:
                if hasattr(self, attr):
                    line = getattr(self, attr)
                    axes[1].plot(line[0], line[1], color)

            if hasattr(self, 'anterior_point'):
                axes[1].scatter(self.anterior_point[0], self.anterior_point[1],
                                s=20, alpha=0.8, label='Anterior', color='green')
            if hasattr(self, 'posterior_point'):
                axes[1].scatter(self.posterior_point[0], self.posterior_point[1],
                                s=20, alpha=0.8, label='Posterior', color='magenta')
            if hasattr(self, 'ap90_points'):
                for p in self.ap90_points:
                    axes[1].scatter(p[0], p[1], s=20, alpha=0.8,
                                    label='AP_perp', color='blue')

            axes[1].set_title('Overlay with Mid Image')
            axes[1].legend()
            fig.tight_layout()
            plt.draw()

        def on_key(event):
            """Handle keyboard events"""
            if event.key == 'm' and mode['current'] == 'view':
                mode['current'] = 'ap_select'
                manual_points.clear()
                ap_perp_points.clear()
                axes[1].clear()
                axes[1].imshow(self.full_embryo_dataset_mid[self.his_channel][0, :, :],
                               cmap='gray')
                axes[1].imshow(self.full_embryo_mask, alpha=0.2)
                axes[1].set_title('Click to select Anterior point, then Posterior point')
                plt.draw()
                print("Manual mode activated. Click to select Anterior point, then Posterior point.")

            elif event.key == 'n' and mode['current'] == 'view':
                if hasattr(self, 'anterior_point') and hasattr(self, 'posterior_point'):
                    self.anterior_point, self.posterior_point = (
                        self.posterior_point, self.anterior_point)
                    # Modify the ap_angle appropriately
                    self.ap_angle = np.pi + self.ap_angle
                    plot_current_state()
                    print("Anterior and Posterior points swapped.")

        def on_click(event):
            """Handle mouse click events"""
            if event.inaxes != axes[1]:
                return

            y, x = event.xdata, event.ydata

            if mode['current'] == 'ap_select':
                self._handle_ap_selection(y, x, manual_points, axes[1], mode,
                                          ap_perp_line_info, plot_current_state)

            elif mode['current'] == 'ap_perp_select':
                self._handle_ap_perp_selection(y, x, ap_perp_points, axes[1],
                                               ap_perp_line_info, mode,
                                               plot_current_state)

        # Connect event handlers and show plot
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('button_press_event', on_click)
        plot_current_state()
        plt.show()

    def _handle_ap_selection(self, y, x, manual_points, ax, mode,
                             ap_perp_line_info, plot_current_state):
        """Handle AP axis point selection"""
        manual_points.append((int(y), int(x)))

        if len(manual_points) == 1:
            # First click - anterior point
            ax.scatter(y, x, s=20, alpha=0.8, color='green',
                       label='Anterior (Manual)')
            ax.set_title('Now click to select Posterior point')
            plt.draw()

        elif len(manual_points) == 2:
            # Second click - posterior point
            ax.scatter(y, x, s=20, alpha=0.8, color='magenta',
                       label='Posterior (Manual)')

            # Update AP measurements
            self.anterior_point = manual_points[0]
            self.posterior_point = manual_points[1]
            dy = self.posterior_point[0] - self.anterior_point[0]
            dx = self.posterior_point[1] - self.anterior_point[1]
            self.ap_angle = np.arctan2(dy, dx)
            print("AP angle: ", np.degrees(self.ap_angle))

            # Draw AP line
            self.ap_line = line(self.anterior_point[0], self.anterior_point[1],
                                self.posterior_point[0], self.posterior_point[1])

            # Calculate AP perpendicular
            ap90_angle = self.ap_angle + np.pi / 2
            mid_y = (self.anterior_point[0] + self.posterior_point[0]) / 2
            mid_x = (self.anterior_point[1] + self.posterior_point[1]) / 2
            mid_point = (mid_y, mid_x)
            self.ap_d = np.hypot(self.anterior_point[1] - self.posterior_point[1],
                                 self.anterior_point[0] - self.posterior_point[0])

            # Draw extended AP90 line
            img_height, img_width = self.full_embryo_mask.shape
            ext_len = min(img_height, img_width)
            ap90_x1 = int(mid_x + ext_len * np.cos(ap90_angle))
            ap90_y1 = int(mid_y + ext_len * np.sin(ap90_angle))
            ap90_x2 = int(mid_x - ext_len * np.cos(ap90_angle))
            ap90_y2 = int(mid_y - ext_len * np.sin(ap90_angle))
            extended_ap90_line = line(ap90_y1, ap90_x1, ap90_y2, ap90_x2)

            # Store line info and update plot
            ap_perp_line_info.update({
                'origin': mid_point,
                'angle': ap90_angle,
                'line_points': [(ap90_y1, ap90_x1), (ap90_y2, ap90_x2)]
            })

            ax.plot([p for p in self.ap_line[0]], [p for p in self.ap_line[1]],
                    'r-', label='AP axis (Manual)')
            ax.plot([p for p in extended_ap90_line[0]],
                    [p for p in extended_ap90_line[1]], 'b--',
                    label='Extended AP90 line')

            # Switch to AP perpendicular selection
            mode['current'] = 'ap_perp_select'
            ax.set_title('Click where AP_perp line intersects embryo boundaries (2 points)')
            ax.legend()
            plt.draw()
            print("AP axis defined. Now click where the perpendicular line intersects the embryo boundaries.")

    def _handle_ap_perp_selection(self, y, x, ap_perp_points, ax,
                                  ap_perp_line_info, mode, plot_current_state):
        """Handle AP perpendicular point selection"""
        click_point = (y, x)
        closest_x, closest_y = self._get_closest_point_on_line(
            click_point,
            ap_perp_line_info['origin'],
            ap_perp_line_info['angle'])

        ap_perp_points.append((int(closest_y), int(closest_x)))
        marker = 'x' if len(ap_perp_points) == 1 else 'o'
        ax.scatter(closest_y, closest_x, s=20, alpha=0.8,
                   color='cyan', marker=marker)

        if len(ap_perp_points) == 1:
            ax.set_title('Click where AP_perp intersects the other side of the embryo')
            plt.draw()

        elif len(ap_perp_points) == 2:
            # Calculate final measurements
            self.ap90_d = np.hypot(ap_perp_points[0][1] - ap_perp_points[1][1],
                                   ap_perp_points[0][0] - ap_perp_points[1][0])
            self.ap90_line = line(ap_perp_points[0][0], ap_perp_points[0][1],
                                  ap_perp_points[1][0], ap_perp_points[1][1])
            self.ap90_points = ap_perp_points

            # Update display
            plot_current_state()
            mode['current'] = 'view'
            self._print_measurements()

    def _get_closest_point_on_line(self, point, origin, angle):
        """Find closest point on line to given point"""
        dx = point[1] - origin[1]
        dy = point[0] - origin[0]
        t = dx * np.cos(angle) + dy * np.sin(angle)
        x = origin[1] + t * np.cos(angle)
        y = origin[0] + t * np.sin(angle)
        return x, y

    def _print_measurements(self):
        """Print AP measurement results"""
        print(f"Manual AP and AP90 axes defined.")
        print(f"Anterior Point: {self.anterior_point}")
        print(f"Posterior Point: {self.posterior_point}")
        print(f"AP Angle: {np.degrees(self.ap_angle):.2f} degrees")
        print(f"AP distance: {self.ap_d:.2f} pixels")
        print(f"AP90 distance: {self.ap90_d:.2f} pixels")

    def _create_ap_points_dataframe(self, save_results):
        """Create and optionally save dataframe of AP measurements"""
        ap_points_df = pd.DataFrame({
            'ap_line': [self.ap_line],
            'ap90_line': [self.ap90_line],
            'ap_d': [self.ap_d],
            'ap90_d': [self.ap90_d],
            'anterior_point': [self.anterior_point],
            'posterior_point': [self.posterior_point],
            'ap90_points': [self.ap90_points],
            'ap_angle': [self.ap_angle],
            'max_loc': [self.max_loc],
            'im_shape_scaled': [self.im_shape_scaled],
            'full_embryo_mask': [self.full_embryo_mask],
        })

        if save_results:
            ap_points_df.to_pickle(os.path.join(self.dataset_folder,
                                                "ap_points_df.pkl"))

        return ap_points_df

    def xy_to_ap(self, compiled_data):
        """Convert xy coordinates to AP axis coordinates"""
        # Get coordinates
        x = compiled_data.x
        y = compiled_data.y

        # Calculate scaling factor
        img_window = self.img_window_dataset[self.his_channel][-1, :, :, :]
        img_window_max = np.max(img_window, axis=0)
        w, h = self.im_shape_scaled
        scale_w = w / img_window_max.shape[1]
        scale_h = h / img_window_max.shape[0]
        scale = np.mean([scale_w, scale_h])

        # Transform coordinates
        scaled_x = scale * x
        scaled_y = scale * y
        full_x = scaled_x + self.max_loc[0]
        full_y = scaled_y + self.max_loc[1]

        # Prepare coordinates for transformation
        xy_pairs = np.transpose(np.stack((full_x, full_y)), (1, 0))
        spots = [np.stack(pair, axis=1) for pair in xy_pairs]

        # Apply transformations
        transform1 = AffineTransform(translation=(-self.anterior_point[0],
                                                  -self.anterior_point[1]))
        transform2 = AffineTransform(rotation=-(self.ap_angle % np.pi))

        # Transform and normalize coordinates
        ap_coords = [transform2(transform1(spot)) / np.array([self.ap_d, self.ap90_d])
                     for spot in spots]

        # Split into AP and AP90 coordinates
        ap = [coords[:, 0] for coords in ap_coords]
        ap90 = [coords[:, 1] for coords in ap_coords]

        # Return updated data
        new_data = compiled_data.copy()
        new_data['ap'] = ap
        new_data['ap90'] = ap90
        return new_data
