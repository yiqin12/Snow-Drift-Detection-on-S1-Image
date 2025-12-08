import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid Qt dependency
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from skimage.filters import threshold_multiotsu
from skimage.draw import ellipse
from skimage.transform import rotate
from matplotlib.patches import Circle
from shapely.geometry import LineString, shape
from rasterio import features


class SnowDrift:
    def __init__(self,
                 input_path,           # Path to GeoJSON/Shapefile
                 output_path=None,
                 n_classes=3,
                 wind_dir=None,        # Wind direction (meteorological degrees, 0° = north wind, clockwise)
                 ROI_idx=0):           # Default to process the 0th ROI


        self.path = None
        self.input_path = input_path
        self.output_path = output_path if output_path is not None else "."
        self.n_classes = n_classes
        self.wind_dir = wind_dir
        self.ROI_idx = ROI_idx

        # Data cache
        self.gdf = None
        self.class1_img = None
        self.transform = None
        self.crs = None
        self.extent = None

        # Computation results
        self.coords = None
        self.p1_geo = None
        self.p2_geo = None
        self.proj_values = None
        self.trail1 = None
        self.trail2 = None

    # ==================== Data Loading ====================
    def load_gdf(self):
        """Load the input GeoDataFrame"""
        self.gdf = gpd.read_file(self.input_path)
        if len(self.gdf) == 0:
            raise ValueError("Input GeoDataFrame is empty")
        return self.gdf

    def load_class1_image(self, raster_path):
        """Load and generate a binary image for class 1 (middle class)"""
        with rasterio.open(raster_path) as src:
            img = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs
            H, W = img.shape
            self.extent = self._compute_extent(W, H)

        # Normalize + multilevel Otsu thresholding
        img_norm = ((img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img) + 1e-9) * 255).astype(np.uint8)
        thresholds = threshold_multiotsu(img_norm, classes=self.n_classes)
        regions = np.digitize(img_norm, bins=thresholds)
        self.class1_img = (regions == 1).astype(np.uint8)  # intermediate class as snowdrift
        # self.class1_img = img
        return self.class1_img


    # ==================== Geometry Processing ====================
    @staticmethod
    def _largest_part(geom):
        """Extract the largest polygon from MultiPolygon"""
        if geom is None:
            return None
        if geom.geom_type == "MultiPolygon":
            return max(geom.geoms, key=lambda g: g.area)
        return geom

    def extract_coords(self):
        """Extract the external contour coordinates of the specified ROI"""
        geom = self._largest_part(self.gdf.iloc[self.ROI_idx].geometry)
        if geom is None or geom.is_empty:
            raise ValueError(f"ROI #{self.ROI_idx} is invalid")
        self.coords = np.array(geom.exterior.coords)[:-1]  # Remove the duplicate closing point
        return self.coords

    def find_wind_perpendicular_extremes(self):
        """
        Find two extreme points of the contour along the direction perpendicular to wind (windward/leeward separation)
        """
        if self.coords is None:
            self.extract_coords()

        wind_rad = np.deg2rad(self.wind_dir)

        # perp_vector = np.array([-np.sin(wind_rad), np.cos(wind_rad)])  # unit vector perpendicular to the wind direction
        # 1. Compute the motion vector of the wind (meteorological → mathematical)
        wind_motion_vector = np.array([
            -np.sin(wind_rad),   # x: east positive; wind from east blows west → x negative
            -np.cos(wind_rad)    # y: north positive; wind from north blows south → y negative
        ])

        # 2. Compute the perpendicular vector (rotate 90° clockwise)
        perp_vector = np.array([
            -wind_motion_vector[1],  # = cos(wind_rad)
            wind_motion_vector[0]    # = -sin(wind_rad)
        ])
        # Normalize
        perp_vector = perp_vector / np.linalg.norm(perp_vector + 1e-9)

        self.proj_values = self.coords @ perp_vector

        idx_max = np.argmax(self.proj_values)
        idx_min = np.argmin(self.proj_values)

        self.p1_geo = tuple(self.coords[idx_min])  # Minimum projection point (e.g., south side)
        self.p2_geo = tuple(self.coords[idx_max])  # Maximum projection point (e.g., north side)

        print(f"📍 Windward boundary point 1: {self.p1_geo} (Projection: {self.proj_values[idx_min]:.2f})")
        print(f"📍 Windward boundary point 2: {self.p2_geo} (Projection: {self.proj_values[idx_max]:.2f})")

        return self.p1_geo, self.p2_geo, self.proj_values

    def compute_perpendicular_separation_length(self, p1=None, p2=None):
        """
        Compute the projected distance between two points perpendicular to the wind direction.

        Args::
            p1, p2: tuple (x, y). Default: self.p1_geo, self.p2_geo

        Returns:
            float: length of the projected segment (in coordinate system units, meters)
        """
        if p1 is None:
            p1 = self.p1_geo
        if p2 is None:
            p2 = self.p2_geo

        if p1 is None or p2 is None:
            raise ValueError("Call find_wind_perpendicular_extremes() first")

        wind_rad = np.deg2rad(self.wind_dir)

        # Wind motion vector (meteorological → mathematical)
        wind_motion_vector = np.array([
            -np.sin(wind_rad),   # x: east positive
            -np.cos(wind_rad)    # y: north positive
        ])

        # Perpendicular unit vector (rotated 90° clockwise)
        perp_vector = np.array([
            -wind_motion_vector[1],  # = cos(wind_rad)
            wind_motion_vector[0]    # = -sin(wind_rad)
        ])
        perp_vector = perp_vector / (np.linalg.norm(perp_vector) + 1e-9)  # Normalize to unit vector

        # Projection
        proj1 = np.dot(p1, perp_vector)
        proj2 = np.dot(p2, perp_vector)

        length = abs(proj2 - proj1)
        print(f"📏 Perpendicular projected separation length: {length:.2f} (units of projection coordinates, typically meters)")

        return length

    # ==================== Snow-Free Trail Detection ====================
    def geo_to_pixel(self, x, y):
        """Convert geographic coordinates to pixel coordinates"""
        col, row = ~self.transform * (x, y)
        return int(round(row)), int(round(col))


    def generate_oriented_ellipse_mask(self, r0, c0, downwind_rad, a, b):
        """
        Generate a filled rotated ellipse mask (high accuracy, no interpolation)
        using geometric calculation: check whether each pixel is inside the rotated ellipse
        """
        H, W = self.class1_img.shape
        mask = np.zeros((H, W), dtype=bool)

        # Create coordinate grids (offset by center)
        rr, cc = np.mgrid[0:H, 0:W]
        rr = rr.astype(np.float64)
        cc = cc.astype(np.float64)

        # Move to ellipse center
        dx = cc - c0
        dy = rr - r0

        # Rotate points counterclockwise by -downwind_rad (reverse-rotate to main axis)
        cos_a = np.cos(-downwind_rad)
        sin_a = np.sin(-downwind_rad)
        x_rot = dx * cos_a + dy * sin_a
        y_rot = -dx * sin_a + dy * cos_a

        # Inside ellipse if (x/a)^2 + (y/b)^2 <= 1
        inside = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0

        mask[inside] = True
        return mask

    def pooling_along_wind_masked(self, start_row, start_col, downwind_rad, a, b,
                                  threshold_snow=0.1, n_consecutive=3, max_steps=200):
        """Perform snow-free trail detection along the wind direction"""
        print(f"🔍 pooling_along_wind_masked received downwind_rad = {np.degrees(downwind_rad):.1f}° (mathematical)")
        self.path = []
        consecutive_snow_count = 0
        r, c = start_row, start_col

        # Generate iceberg mask (once only)
        geom = self._largest_part(self.gdf.iloc[self.ROI_idx].geometry)
        H, W = self.class1_img.shape
        iceberg_mask = features.rasterize(
            [(geom, 1)],
            out_shape=(H, W),
            transform=self.transform,
            fill=0,
            default_value=1,
            all_touched=True
        ).astype(bool)

        for step in range(max_steps):
            if not (0 <= r < H and 0 <= c < W):
                break

            ellipse_mask = self.generate_oriented_ellipse_mask(r, c, downwind_rad, a, b)
            valid_mask = ellipse_mask & (~iceberg_mask) & (self.class1_img > 0)

            coords = np.where(valid_mask)
            kernel_points = list(zip(coords[0], coords[1]))

            if len(kernel_points) == 0:
                break

            values = [self.class1_img[r_p, c_p] for r_p, c_p in kernel_points]
            mean_val = np.mean(values)
            is_snow = mean_val < threshold_snow

            if is_snow:
                consecutive_snow_count += 1
                if consecutive_snow_count >= n_consecutive:
                    self.path.append((r, c))
                    break
            else:
                consecutive_snow_count = 0

            self.path.append((r, c))

            # Move along the wind direction (image coordinates: row increases downward)）
            # r += int(round(np.sin(downwind_rad)))  
            # c += int(round(np.cos(downwind_rad)))
            r += -np.sin(downwind_rad)  #  negative sign makes upward movement decrease row index
            c += np.cos(downwind_rad)

        return self.path

    def detect_snow_free_trails(self,
                                raster_path,
                                kernel_length=21,
                                kernel_width=5,
                                threshold_snow=0.1,
                                n_consecutive=3,
                                max_steps=200):
        """
         Main procedure for detecting snow-free trails
        """
        self.load_class1_image(raster_path)
        self.load_gdf()
        self.find_wind_perpendicular_extremes()

        # Downwind direction (mathematical angle: 0°=east, counterclockwise)
        downwind_direction_deg = (self.wind_dir + 180) % 360
        downwind_rad = np.radians(90 - downwind_direction_deg)  # Meteorological → mathematical

        a = kernel_length // 2
        b = kernel_width // 2

        r1, c1 = self.geo_to_pixel(*self.p1_geo)
        r2, c2 = self.geo_to_pixel(*self.p2_geo)

        path1 = self.pooling_along_wind_masked(r1, c1, downwind_rad, a, b, threshold_snow, n_consecutive, max_steps)
        path2 = self.pooling_along_wind_masked(r2, c2, downwind_rad, a, b, threshold_snow, n_consecutive, max_steps)

        def pixels_to_geo(path):
            return [self.transform * (c, r) for r, c in path]

        geo_path1 = pixels_to_geo(path1)
        geo_path2 = pixels_to_geo(path2)

        self.trail1 = LineString(geo_path1) if len(geo_path1) > 1 else None
        self.trail2 = LineString(geo_path2) if len(geo_path2) > 1 else None

        return {
            "trail1": self.trail1,
            "trail2": self.trail2,
            "start_points": (self.p1_geo, self.p2_geo),
            "downwind_direction_deg": downwind_direction_deg,
            "path1_pixels": path1,
            "path2_pixels": path2
        }

    def compute_trail_lengths(self):
        """
        Compute the geographical lengths of both trails (in meters)

        Returns:
            dict: containing trail1 and trail2 lengths (m)
        """
        lengths = {}

        if self.trail1 is not None and not self.trail1.is_empty:
            lengths["trail1_length_m"] = self.trail1.length
        else:
            lengths["trail1_length_m"] = 0.0

        if self.trail2 is not None and not self.trail2.is_empty:
            lengths["trail2_length_m"] = self.trail2.length
        else:
            lengths["trail2_length_m"] = 0.0

        print(f"📐 Trail 1 Length: {lengths['trail1_length_m']:.2f} m")
        print(f"📐 Trail 2 Length: {lengths['trail2_length_m']:.2f} m")

        return lengths

    # ==================== Visualization ====================
    def plot_snow_free_trail(self,
                             raster_path,
                             kernel_length=21,
                             kernel_width=5,
                             threshold_snow=0.1,
                             n_consecutive=3,
                             max_steps=200,
                             show_kernel=False,
                             dpi=150,
                             save_path=None):
        """
        Visualize the detected snow-free trail results
        """
        results = self.detect_snow_free_trails(raster_path, kernel_length, kernel_width,
                                               threshold_snow, n_consecutive, max_steps)

        fig, ax = plt.subplots(figsize=(14, 12), dpi=dpi)
        ax.imshow(self.class1_img, cmap="gray", extent=self.extent, alpha=0.7)

        # Iceberg outline
        geom = self._largest_part(self.gdf.iloc[self.ROI_idx].geometry)
        gpd.GeoSeries([geom]).plot(ax=ax, facecolor='red', edgecolor='red', linewidth=2, alpha=0.3)

        # Iceberg outline
        if self.trail1:
            gpd.GeoSeries([self.trail1]).plot(ax=ax, color='cyan', linewidth=2.5, label='Trail Edge 1')
        if self.trail2:
            gpd.GeoSeries([self.trail2]).plot(ax=ax, color='cyan', linewidth=2.5, label='Trail Edge 2')

        # Starting points
        ax.scatter(self.p1_geo[0], self.p1_geo[1], c='red', s=100, marker='x', linewidth=3, label='Start Points')
        ax.scatter(self.p2_geo[0], self.p2_geo[1], c='red', s=100, marker='x', linewidth=3)

        # Wind direction arrow
        center_x, center_y = geom.centroid.x, geom.centroid.y
        downwind_direction_deg = results["downwind_direction_deg"]
        downwind_rad = np.radians(90 - downwind_direction_deg)
        dx = 1500 * np.cos(downwind_rad)
        dy = 1500 * np.sin(downwind_rad)
        ax.arrow(center_x, center_y, dx, dy,
                 width=15, head_width=50, head_length=80,
                 color='blue', alpha=0.9, linewidth=2, label=f'Wind → {downwind_direction_deg}°')

        # Optional: display the first sampling kernel
        if show_kernel and len(results["path1_pixels"]) > 0:
            r1, c1 = results["path1_pixels"][0]
            ellipse_mask = self.generate_oriented_ellipse_mask(r1, c1, downwind_rad, kernel_length//2, kernel_width//2)
            valid_mask = ellipse_mask & (~features.rasterize([(geom, 1)], out_shape=self.class1_img.shape,
                                                            transform=self.transform, fill=0, all_touched=True).astype(bool))
            coords = np.where(valid_mask)
            if len(coords[0]) > 0:
                ax.scatter(coords[1], coords[0], c='yellow', s=4, marker='s', alpha=0.8, label='Sampling Kernel')

        # Configure figure
        ax.set_aspect("equal")
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title(f"ROI #{self.ROI_idx}: Wind-Guided Snow-Free Trail Detection\n"
                     f"Wind from {self.wind_dir}° → Downwind {downwind_direction_deg}° | "
                     f"Kernel: {kernel_length}×{kernel_width}", fontsize=14)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.3)

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.show()

        return results

    def plot_roi_local(
        self,
        raster_path=None,
        pad_x=400,          # horizontal padding (in pixels)
        pad_y=100,          # vertical padding (in pixels)
        face_alpha=0.25,
        show_trails=True,   # whether to show trails
        show_start_points=True,
        show_wind_arrow=True,
        show_first_kernel=False,  # whether to show the first sampling kernel
        dpi=150,
        save_path=None
    ):
        """
        Local visualization of the current ROI, including trails, wind direction, and start points
        """
        if self.gdf is None:
            self.load_gdf()

        geom = self._largest_part(self.gdf.iloc[self.ROI_idx].geometry)
        if geom is None or geom.is_empty:
            raise ValueError(f"ROI #{self.ROI_idx} 无效")

        # Load base image if not already loaded
        if self.class1_img is None:
            if raster_path is None:
                raise ValueError("Please provide raster_path or call detect_snow_free_trails first")
            self.load_class1_image(raster_path)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

        # Display class1 background image
        ax.imshow(self.class1_img, cmap="gray", extent=self.extent, alpha=0.8)

        # Convert padding from pixels to meters
        pixel_size_x = abs(self.transform.a)
        pixel_size_y = abs(self.transform.e)
        pad_x_meters = pad_x * pixel_size_x
        pad_y_meters = pad_y * pixel_size_y

        # All other ROI boundaries (light color)
        try:
            self.gdf.boundary.plot(ax=ax, color=(0,1,0,0.2), linewidth=0.6)
        except Exception:
            pass

        # Current ROI: red fill
        gpd.GeoSeries([geom]).plot(ax=ax, facecolor=(1,0,0,face_alpha), edgecolor="red", linewidth=2.0)

        # ========== Snowdrift-specific elements ==========

        # 1. Trail lines
        if show_trails:
            if hasattr(self, 'trail1') and self.trail1 is not None:
                gpd.GeoSeries([self.trail1]).plot(ax=ax, color='cyan', linewidth=2.5, label='Trail Edge 1')
            if hasattr(self, 'trail2') and self.trail2 is not None:
                gpd.GeoSeries([self.trail2]).plot(ax=ax, color='cyan', linewidth=2.5, label='Trail Edge 2')

        # 2. Starting points
        if show_start_points and hasattr(self, 'p1_geo') and hasattr(self, 'p2_geo'):
            ax.scatter(self.p1_geo[0], self.p1_geo[1], c='red', s=100, marker='x', linewidth=3, label='Start Points')
            ax.scatter(self.p2_geo[0], self.p2_geo[1], c='red', s=100, marker='x', linewidth=3)

        # 3. Wind direction arrow
        if show_wind_arrow and self.wind_dir is not None:
            center_x, center_y = geom.centroid.x, geom.centroid.y
            downwind_direction_deg = (self.wind_dir + 180) % 360
            downwind_rad = np.radians(90 - downwind_direction_deg)
            dx = 1500 * np.cos(downwind_rad)
            dy = 1500 * np.sin(downwind_rad)
            ax.arrow(center_x, center_y, dx, dy,
                     width=15, head_width=50, head_length=80,
                     color='blue', alpha=0.9, linewidth=2, label=f'Wind → {downwind_direction_deg}°')

        # 4. First sampling kernel (debug visualization)
        if show_first_kernel and hasattr(self, 'path1_pixels') and len(self.path1_pixels) > 0:
            r1, c1 = self.path1_pixels[0]
            # Generate local kernel mask
            a = 10  # Example: kernel_length=21 → a=10
            b = 2   # Example: kernel_width=5 → b=2
            ellipse_mask = self.generate_oriented_ellipse_mask(r1, c1, downwind_rad, a, b, max_radius=50)
            # Convert to non-iceberg area
            iceberg_mask = features.rasterize(
                [(geom, 1)],
                out_shape=self.class1_img.shape,
                transform=self.transform,
                fill=0,
                default_value=1,
                all_touched=True
            ).astype(bool)
            valid_mask = ellipse_mask & (~iceberg_mask)
            coords = np.where(valid_mask)
            if len(coords[0]) > 0:
                # Convert to geocoordinate
                geo_coords = [self.transform * (c, r) for r, c in zip(coords[0], coords[1])]
                if geo_coords:
                    xs, ys = zip(*geo_coords)
                    ax.scatter(xs, ys, c='yellow', s=4, marker='s', alpha=0.8, label='Sampling Kernel')

        # ========== Scaling and annotations ==========

        xmin, ymin, xmax, ymax = geom.bounds
        ax.set_xlim(xmin - pad_x_meters, xmax + pad_x_meters)
        ax.set_ylim(ymin - pad_y_meters, ymax + pad_y_meters)

        # Shape metrics
        M = self.roi_metrics()
        ax.set_title(f"ROI #{self.ROI_idx} | Area={M['area_m2']:.0f}m²  Peri={M['perimeter_m']:.1f}m  "
                     f"Long={M['long_axis_m']:.1f}m  Short={M['short_axis_m']:.1f}m  "
                     f"Elong={M['elongation']:.2f}\n"
                     f"Wind: {self.wind_dir}° → Trails: {getattr(self, 'trail_length1', 0):.0f}m, {getattr(self, 'trail_length2', 0):.0f}m",
                     fontsize=10)
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.show()

        return M

    # ==================== Utility Methods ====================
    def _compute_extent(self, width, height):
        """Compute raster extent"""
        x0, y0 = self.transform * (0, 0)
        x1, y1 = self.transform * (width, height)
        return (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

    def roi_metrics(self):
        """Compute geometric metrics of the current ROI"""
        geom = self._largest_part(self.gdf.iloc[self.ROI_idx].geometry)
        area = float(geom.area)
        per = float(geom.length)

        rect = geom.minimum_rotated_rectangle
        x, y = rect.exterior.coords.xy
        edges = np.hypot(np.diff(x), np.diff(y))[:-1]
        long_axis = float(edges.max()) if len(edges) >= 2 else 0.0
        short_axis = float(edges.min()) if len(edges) >= 2 else 0.0
        elong = long_axis / (short_axis + 1e-9)

        return {
            "area_m2": area,
            "perimeter_m": per,
            "long_axis_m": long_axis,
            "short_axis_m": short_axis,
            "elongation": elong,
            "bounds": geom.bounds
        }

    def save_results(self, filename_prefix="snowdrift"):
        """Save the detected snow-free trail results"""
        import os
        os.makedirs(self.output_path, exist_ok=True)

        if self.trail1 or self.trail2:
            trails = []
            if self.trail1: trails.append(self.trail1)
            if self.trail2: trails.append(self.trail2)
            gdf_trails = gpd.GeoDataFrame({"type": ["trail1", "trail2"][:len(trails)]}, geometry=trails, crs=self.crs)
            geojson_path = os.path.join(self.output_path, f"{filename_prefix}_trails.geojson")
            gdf_trails.to_file(geojson_path, driver="GeoJSON")

            print(f"Trail results saved to: {geojson_path}")
