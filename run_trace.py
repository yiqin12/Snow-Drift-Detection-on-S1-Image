import os
import re
import glob
import numpy as np
import csv
from SnowDrift import SnowDrift

ROI_idx_all = np.array([[13, 15, 16, 18, 22, 25],
                        [14, 16, 17, 20, 23, 25],
                        [14, 16, 18, 20, 23, 25],
                        [16, 18, 19, 22, 25, 30],
                        [13, 15, 16, 18, 21, 23],
                        [16, 18, 19, 20, 23, None],
                        [31, 33, 34, 36, 40, 44],
                        [15, 17, 18, 20, 23, 25],
                        [15, 17, 19, 21, 24, 26],
                        [15, 17, 18, 20, 24, 27],
                        [14, 16, 17, 19, 22, 24],
                        [13, 15, 16, 18, 22, 26],
                        [14, 16, 17, 19, 22, 24],
                        [102, 104, 106, 108, 111, None],
                        [13, 15, 16, 18, 23, 27],
                        [15, 17, 18, 20, 23, 25], [17, 19, 20, 21, 25, 28], [13, 15, 16, 18, 21, 23],
                        [16, 18, 20, 22, 25, 27], [15, 17, 18, 20, 24, 28], [20, None, None, None, None, 31],
                        [13, 15, 16, 17, 20, None], [28, 44, 45, 48, 53, 54], [17, 19, 20, 22, 25, 27],
                        [31, 33, 34, 35, 39, 41], [45, 47, 48, 49, 53, None], [17, 20, 22, 23, 26, 28],
                        [30, 32, 33, 34, 38, 41],
                        [13, 15, 16, 18, 21, 23], [14, 16, 18, 20, 23, 25], [14, 16, 17, 19, 23, 26],
                        [11, 13, 14, 16, 19, 21], [13, 15, 17, 19, 22, 24], [21, 27, 28, 30, 34, 36],
                        [24, 39, 40, 41, 45, None],
                        [39, 48, 49, 50, 53, None], [18, 30, 31, 32, 35, None], [34, 44, 49, None, 60, None],
                        [20, 30, 34, 35, 39, None], [27, 33, 37, 47, 51, None]
                        ])

ROI_idx_all = ROI_idx_all[-20::, :]

# === Directory settings ===
raster_dir = r"G:\SD\S1_AtkaBay_2022_sigma0_dB"  # directory containing raster images
processed_dir = r"G:\SD\processed"  # daily output folders (output_YYYYMMDD)
final_dir = r"G:\SD\final_SD"  # final output folder for daily results and summary
os.makedirs(final_dir, exist_ok=True)

# === Collect images and extract acquisition dates ===
tifs = sorted(glob.glob(os.path.join(raster_dir, "S1_AtkaBay_*_HH_sigma0_dB.tif")))
date_pat = re.compile(r"(\d{8})")
day_paths = []
for p in tifs:
    m = date_pat.search(os.path.basename(p))
    if m:
        day_paths.append((m.group(1), p))  # (YYYYMMDD, tif_path)
day_paths.sort(key=lambda x: x[0])

# === Match number of days between image list and ROI table ===
n_days_imgs = len(day_paths)
n_days_rois = ROI_idx_all.shape[0]
n_days = min(n_days_imgs, n_days_rois)
length_all = np.full((n_days, 6, 2), np.nan, dtype=float)

print(f"Matched {n_days} days（Images: {n_days_imgs}，ROI table: {n_days_rois}）。")

for day_i in range(n_days):
    day_str, tif_path = day_paths[day_i]  # YYYYMMDD, image of the current day
    polygons_geojson = os.path.join(processed_dir, f"output_{day_str}", "class2_polygons.geojson")
    if not os.path.isfile(polygons_geojson):
        print(f"[Skipped] Missing polygon file:{polygons_geojson}")
        continue

    # Six ROIs per day (may contain None)
    rois = list(ROI_idx_all[day_i])

    print(f"\n=== {day_str} ===")
    # Create an output subfolder for the current image
    day_out_dir = os.path.join(final_dir, day_str)
    os.makedirs(day_out_dir, exist_ok=True)

    for j in range(6):
        roi = rois[j]
        if roi is None or (isinstance(roi, float) and np.isnan(roi)):
            print(f"  ROI[{j}] = None, skipped.")
            continue

        try:
            roi_int = int(roi)
        except Exception:
            print(f"  ROI[{j}] = {roi} is not an integer, skipped.")
            continue

        try:
            # —— SnowDrift workflow  ——
            sd = SnowDrift(
                input_path=polygons_geojson,  # F:\Sd\processed\output_{YYYYMMDD}\class2_polygons.geojson
                wind_dir=90,
                ROI_idx=roi_int
            )

            sd.load_gdf()
            sd.extract_coords()
            sd.find_wind_perpendicular_extremes()
            _ = sd.compute_perpendicular_separation_length()

            # Save result plot to F:\Sd\final_SD\{YYYYMMDD}\ROI{roi}snowdrift_trail.png
            save_png = os.path.join(day_out_dir, f"ROI{roi_int}snowdrift_trail.png")
            _ = sd.plot_snow_free_trail(
                raster_path=tif_path,  # F:\Sd\S1_AtkaBay_2022_sigma0_dB\...
                kernel_length=15,
                kernel_width=11,
                threshold_snow=0.6,
                n_consecutive=5,
                max_steps=2000,
                show_kernel=True,
                save_path=save_png
            )

            t1 = getattr(getattr(sd, "trail1", None), "length", np.nan)
            t2 = getattr(getattr(sd, "trail2", None), "length", np.nan)
            length_all[day_i, j, 0] = t1
            length_all[day_i, j, 1] = t2

            print(f"  ROI{roi_int}: trail1={t1:.3f}, trail2={t2:.3f} -> {os.path.basename(save_png)}")

        except Exception as e:
            print(f"  [Error] {day_str} ROI{roi_int} failed to process: {e}")
            # Continue to next ROI

# === Save summary results ===
npy_path = os.path.join(final_dir, "length_all.npy")
np.save(npy_path, length_all)
print(f"\n[Saved] Length matrix -> {npy_path}")

csv_path = os.path.join(final_dir, "length_all.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["day", "slot(0-5)", "ROI_idx", "trail1_length", "trail2_length"])
    for day_i in range(n_days):
        day_str = day_paths[day_i][0]
        rois = list(ROI_idx_all[day_i])
        for j in range(6):
            roi = rois[j]
            roi_out = "" if (roi is None or (isinstance(roi, float) and np.isnan(roi))) else int(roi)
            t1 = length_all[day_i, j, 0]
            t2 = length_all[day_i, j, 1]
            w.writerow([day_str, j, roi_out, "" if np.isnan(t1) else t1, "" if np.isnan(t2) else t2])

print(f"[Saved] Detailed CSV -> {csv_path}")
