import re, sys, os
import tifffile

path = sys.argv[1] if len(sys.argv) > 1 else '_1_MMStack_Default.ome.tif'

# is_mmstack=False / is_ome=False skip the broken index-map parsing
with tifffile.TiffFile(path, is_mmstack=False, is_ome=False) as tf:
    p0 = tf.pages[0]
    print("file size (GB):", os.path.getsize(path) / 1e9)
    print("pages in this file:", len(tf.pages))
    print("frame shape (H, W):", p0.shape, " dtype:", p0.dtype, " bits:", p0.bitspersample)

    desc = p0.description or ""
    print("\n--- OME-XML highlights (describes the WHOLE acquisition) ---")
    for key in ["SizeX", "SizeY", "SizeT", "SizeC", "SizeZ", "PhysicalSizeX",
                "TimeIncrement", "DimensionOrder", "Type"]:
        m = re.search(rf'{key}="([^"]+)"', desc)
        if m:
            print(key, "=", m.group(1))
    print("UUID files referenced:", re.findall(r'FileName="([^"]+)"', desc)[:10])

    # Micro-Manager summary metadata, if readable
    for name in ("MicroManagerMetadata",):
        tag = p0.tags.get(name)
        if tag:
            print("\nMM per-frame metadata:", str(tag.value)[:1500])

    t = [tf.pages[i].tags["MicroManagerMetadata"].value["ElapsedTime-ms"] for i in (0, 1, 2, 3000, 6045)]
    print(t)