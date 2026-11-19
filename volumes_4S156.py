"""volumes_4S156.py - regional volumes (mm3) of the 156 regions of the 4S156 atlas.

For each participant the number of voxels per region in the subject-space atlas
segmentation (dseg) is counted and multiplied by the voxel volume.
Output: ~/volumes_4S156.csv (one row per participant, columns r1 ... r156). Requires nibabel."""

import glob, os, re
import numpy as np, nibabel as nib, scipy.io as sio, pandas as pd

BASE = os.environ.get("SNET_BASE", "snet_data")
root = os.environ.get("SNET_SEG_DIR", f"{BASE}/qsirecon_segmentations")   # subject-space dseg files
mat_root = os.environ.get("SNET_QSIRECON_DIR", f"{BASE}/qsirecon_outputs")   # connectivity matrices (.mat)
seg_files = sorted(glob.glob(f"{root}/sub-*/**/*space-ACPC_seg-4S156Parcels_dseg.nii.gz", recursive=True))
print("dseg files:", len(seg_files))

# region order: region_ids in the .mat file (same order as the matrix index)
example = sorted(glob.glob(f"{mat_root}/sub-*/4S156Parcels_radius2_count_connectivity.mat"))[0]
ids = sio.loadmat(example)["atlas_4S156Parcels_region_ids"].ravel().astype(int)
print("number of regions:", len(ids), "first 5 ids:", ids[:5])

rows = []
for f in seg_files:
    sub = re.search(r"(sub-[0-9A-Za-z]+)", f).group(1)
    img = nib.load(f)
    data = np.asanyarray(img.dataobj).astype(int)
    voxel_mm3 = float(np.prod(img.header.get_zooms()[:3]))
    counts = {i: int((data == i).sum()) for i in ids}
    row = {"subject": sub, "voxel_mm3": voxel_mm3}
    row.update({f"r{i}": counts[i] * voxel_mm3 for i in ids})
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(os.path.expanduser("~/volumes_4S156.csv"), index=False)
print("written:", df.shape)
vol_only = df[[c for c in df.columns if c.startswith("r")]]
print("empty regions (volume=0):", int((vol_only == 0).sum().sum()))
print("min/median/max volume mm3:", vol_only.values.min(), np.median(vol_only.values), vol_only.values.max())
