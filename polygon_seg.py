import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import shape
from shapely.ops import unary_union
from skimage.filters import threshold_multiotsu
from skimage.morphology import remove_small_objects, remove_small_holes, opening, closing, disk
import matplotlib.pyplot as plt


class polygon_seg:
    def __init__(self,
                 input_path,
                 output_path=None,
                 n_classes=3,
                 min_area_m2=5000,
                 hole_area_m2=2000,
                 open_radius_m=20,
                 close_radius_m=0,
                 buffer_m=25.0,
                 simplify_m=2.0,
                 keep_area_min_m2=10000,
                 small_area_thr_m2=20000,
                 compact_max_for_small=0.70,
                 elong_min=None):

        self.input_path = input_path
        self.output_path = output_path if output_path is not None else "."
        self.n_classes = n_classes

        # Image data
        self.src = None
        self.img = None
        self.transform = None
        self.crs = None
        self.H = None
        self.W = None

        # Segmentation results
        self.thresholds = None
        self.regions = None
        self.masks = []
        self.target_mask = None
        self.gdf = None

        # clean_binary_mask parameters
        self.min_area_m2 = min_area_m2
        self.hole_area_m2 = hole_area_m2
        self.open_radius_m = open_radius_m
        self.close_radius_m = close_radius_m

        # smooth_polygons parameters
        self.buffer_m = buffer_m
        self.simplify_m = simplify_m

        # polygon filtering parameters
        self.keep_area_min_m2 = keep_area_min_m2
        self.small_area_thr_m2 = small_area_thr_m2
        self.compact_max_for_small = compact_max_for_small
        self.elong_min = elong_min

    def load_and_segment(self):
        """Load raster image and excute segmentation"""
        with rasterio.open(self.input_path) as src:
            self.src = src
            self.img = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs

        self.H, self.W = self.img.shape

        # Exclude NaN/Inf values before thresholding
        valid = np.isfinite(self.img)
        vals = self.img[valid]
        self.thresholds = threshold_multiotsu(vals, classes=self.n_classes)
        self.regions = np.digitize(self.img, bins=self.thresholds)
        self.masks = [(self.regions == i).astype(np.uint8) for i in range(self.n_classes)]

        return self.masks

    def polygonize(self, target_class=-1):
        """
        Polygonization workflow
        target_class: index of the target class, default -1 means the last one
        """
        # Set the target mask
        if target_class >= 0 and target_class < len(self.masks):
            self.target_mask = self.masks[target_class]
        else:
            # default: the last one as the reference to segment the iceberg polygons
            self.target_mask = self.masks[-1]

        # 3) Mask cleaning (remove small objects/spikes)
        mask_clean = self.clean_binary_mask()

        # 4) Polygonization
        self.gdf = self.polygonize_mask(mask_clean)

        if len(self.gdf) == 0:
            print("No polygons after cleaning, try lowering thresholds")
            return self.gdf

        # 5) Add shape metrics
        self.gdf = self.add_shape_metrics()

        # 6) Filter by area
        self.gdf = self.gdf[self.gdf["area_m2"] >= self.keep_area_min_m2]

        # 7) Remove small & compact polygons
        if self.small_area_thr_m2 and self.compact_max_for_small:
            condition = ~((self.gdf["area_m2"] < self.small_area_thr_m2) &
                          (self.gdf["compact"] > self.compact_max_for_small))
            self.gdf = self.gdf[condition]

        # 8) Elongation filtering
        if self.elong_min is not None:
            self.gdf = self.gdf[self.gdf["elong"] >= float(self.elong_min)]

        # 9) Smooth polygon boundaries
        self.gdf = self.smooth_polygons()

        return self.gdf

    def compute_extent(self):
        """Compute imshow extent from affine transform and dimensions"""
        if self.transform is None:
            raise ValueError("Please call load_and_segment() first")

        x0, y0 = self.transform * (0, 0)
        x1, y1 = self.transform * (self.W, self.H)
        return (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

    def clean_binary_mask(self, mask=None,
                      min_area_m2=5000, hole_area_m2=2000,
                      open_radius_m=20, close_radius_m=0):
        """Mask cleaning with parameters in 'meters' (remove small objects/holes + morphological open/close)"""
        if mask is None:
            mask = self.target_mask

        if self.transform is None:
            raise ValueError("Please call load_and_segment() first")

        px = float(abs(self.transform.a))  # pixel size (m)
        px_area = px * px
        min_pixels = int(round(min_area_m2 / px_area))
        hole_pixels = int(round(hole_area_m2 / px_area))
        r_open_px = max(1, int(round(open_radius_m / px))) if open_radius_m > 0 else 0
        r_close_px = max(1, int(round(close_radius_m / px))) if close_radius_m > 0 else 0

        m = mask.astype(bool)

        if min_pixels > 0:
            m = remove_small_objects(m, min_size=min_pixels)
        if hole_pixels > 0:
            m = remove_small_holes(m, area_threshold=hole_pixels)
        if r_open_px:
            m = opening(m, disk(r_open_px))
        if r_close_px:
            m = closing(m, disk(r_close_px))
        return m.astype(np.uint8)

    def polygonize_mask(self, mask):
        """rasterio.features.shapes → GeoDataFrame"""
        if self.transform is None or self.crs is None:
            raise ValueError("Please call load_and_segment() first")

        geoms = []
        for geom, val in features.shapes(mask, mask=mask, transform=self.transform):
            if val != 1:
                continue
            poly = shape(geom)
            if not poly.is_valid:
                poly = poly.buffer(0)
            geoms.append(poly)

        if len(geoms) == 0:
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)

        gdf = gpd.GeoDataFrame({"class": [f"class{len(self.masks) - 1}"] * len(geoms)},
                               geometry=geoms, crs=self.crs)
        return gdf

    def add_shape_metrics(self, gdf=None):
        """Add area, perimeter, compactness, elongation metrics"""
        if gdf is None:
            gdf = self.gdf

        if gdf is None or len(gdf) == 0:
            return gdf

        gdf = gdf.copy()
        gdf["area_m2"] = gdf.area
        gdf["peri_m"] = gdf.length
        gdf["compact"] = 4 * np.pi * gdf["area_m2"] / (gdf["peri_m"] ** 2 + 1e-9)  # circle=1

        def elongation(geom):
            try:
                rect = geom.minimum_rotated_rectangle
                if rect.is_empty:
                    return 0.0
                x, y = rect.exterior.coords.xy
                edges = np.hypot(np.diff(x), np.diff(y))[:-1]
                if len(edges) >= 2:
                    major, minor = np.sort(edges)[-2:]
                    return float(major / (minor + 1e-9))
                else:
                    return 0.0
            except:
                return 0.0

        gdf["elong"] = gdf.geometry.apply(elongation)
        return gdf

    def smooth_polygons(self, gdf=None, buffer_m=2.0, simplify_m=2.0):
        """Mild smoothing：buffer(+r/-r)+simplify"""
        if gdf is None:
            gdf = self.gdf

        if gdf is None or len(gdf) == 0:
            return gdf

        if buffer_m and buffer_m > 0:
            gdf = gdf.copy()
            try:
                gdf["geometry"] = gdf.buffer(buffer_m).buffer(-buffer_m)
            except:
                print("Buffer operation failed, skipping smoothing")

        if simplify_m and simplify_m > 0:
            gdf = gdf.copy()
            try:
                gdf["geometry"] = gdf.simplify(simplify_m, preserve_topology=True)
            except:
                print("Simplification failed, skipping")

        return gdf

    def get_results(self):
        """Return processed results"""
        return {
            'image': self.img,
            'masks': self.masks,
            'polygons': self.gdf,
            'thresholds': self.thresholds
        }

    def save_results(self, filename_prefix="result"):
        """Save results to GeoJSON and Shapefile"""
        import os
        os.makedirs(self.output_path, exist_ok=True)

        if self.gdf is not None and len(self.gdf) > 0:
            # Save GeoJSON
            geojson_path = os.path.join(self.output_path, f"{filename_prefix}.geojson")
            self.gdf.to_file(geojson_path, driver="GeoJSON")

            # Save Shapefile
            shp_path = os.path.join(self.output_path, f"{filename_prefix}.shp")
            self.gdf.to_file(shp_path)

            print(f"Results saved to: {geojson_path} and {shp_path}")

    def plot_all_rois_on_raster(
        self,
        raster_path: str = None,        # default: self.input_path
        gdf: gpd.GeoDataFrame = None,   # default: self.gdf
        fill_alpha: float = 0.22,
        edgecolor: str = "lime",
        linewidth: float = 1.0,
        label_offset_px=(6, 6),
        label_color="white",
        label_bg=True,
        label_min_area_m2: float = 0.0,
        start_index: int = 0,
        dpi: int = 150,
        save_path: str = None
    ):
        """
        Overlay the current or given ROI polygons on the original raster,
        with index labels.
        """
        # Use class data as defaults
        if gdf is None:
            gdf = self.gdf
        if raster_path is None:
            raster_path = self.input_path

        if gdf is None or len(gdf) == 0:
            raise ValueError("gdf is empty, no ROI to plot.")

        # Read raster base map
        with rasterio.open(raster_path) as src:
            img = src.read(1)
            extent = self._extent(src.transform, src.width, src.height)
            r_crs = src.crs

        # CRS alignment with raster
        gdf_plot = gdf if (not gdf.crs or gdf.crs == r_crs) else gdf.to_crs(r_crs)

        # Plot: raster as background
        fig, ax = plt.subplots(figsize=(10,10), dpi=dpi)
        ax.imshow(img, cmap="gray", extent=extent, zorder=1)
        ax.set_aspect("equal", adjustable="box")

        # Plot ROIs (semi-transparent fill + edge line)
        gdf_plot.plot(ax=ax, facecolor=(0,1,0,fill_alpha), edgecolor=edgecolor, linewidth=linewidth, zorder=2)

        # Add labels
        for i, geom in enumerate(gdf_plot.geometry):
            if geom is None or geom.is_empty:
                continue
            geom = self._largest_part(geom)
            if label_min_area_m2 and geom.area < label_min_area_m2:
                continue
            p = geom.representative_point()
            ax.annotate(
                str(start_index + i),
                (p.x, p.y), xytext=label_offset_px, textcoords="offset points",
                color=label_color, fontsize=8,
                bbox=(dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.55) if label_bg else None),
                zorder=3
            )

        # Keep full image view
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title("All ROIs overlaid on original raster (with indices)")
        ax.set_xlabel("Easting"); ax.set_ylabel("Northing")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.show()

    @staticmethod
    def _largest_part(geom):
        if geom is None:
            return None
        if geom.geom_type == "MultiPolygon":
            return max(geom.geoms, key=lambda g: g.area)
        return geom

    @staticmethod
    def _extent(transform, w, h):
        x0, y0 = transform * (0, 0)
        x1, y1 = transform * (w, h)
        return (min(x0,x1), max(x0,x1), min(y0,y1), max(y0,y1))