import os
import re
import glob
from polygon_seg import polygon_seg

# Root directory (folder containing the .tif files)
input_dir = r"G:\SD\S1_AtkaBay_2022_sigma0_dB"

# Scan all .tif files
tif_list = sorted(glob.glob(os.path.join(input_dir, "S1_AtkaBay_*_HH_sigma0_dB.tif")))

# Batch processing loop — without defining a separate function
for tif_path in tif_list:
    fname = os.path.basename(tif_path)
    #  Extract date (YYYYMMDD) from filename
    m = re.search(r"\d{8}", fname)
    if not m:
        print(f"[Skipped] No date found in filename: {fname}")
        continue
    day1 = m.group(0)  # e.g. '20221104'

    # Create an output folder for the current image
    output_path = rf"G:\SD\processed\output_{day1}"
    os.makedirs(output_path, exist_ok=True)

    print(f"\n=== Processing: {fname} | Date: {day1} ===")
    try:
        # Create instance
        seg = polygon_seg(
            input_path=tif_path,
            output_path=output_path,
            min_area_m2=10000,
            hole_area_m2=4000,
            open_radius_m=30,
            close_radius_m=0,
            keep_area_min_m2=10000,
            small_area_thr_m2=20000,
            compact_max_for_small=0.70,
            elong_min=None,
            buffer_m=25.0,
            simplify_m=2.0
        )

        # Load and segment
        masks = seg.load_and_segment()

        # In most cases, polygonize() processes the “last class” by default
        polygons = seg.polygonize()

        # Overlay visualization (saved to corresponding output directory)
        seg.plot_all_rois_on_raster(
            linewidth=1.0,
            label_min_area_m2=0,
            start_index=0,
            save_path=os.path.join(output_path, "output_rois_plot.png")
        )

        # Retrieve and save results
        results = seg.get_results()
        seg.save_results("class2_polygons")  # Usually saved in the same output_path

        print(f"[Done] {fname} → Output directory:: {output_path}")

    except Exception as e:
        print(f"[Error] Failed to process {fname} ：{e}")
