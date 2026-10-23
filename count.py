import json, collections, re

PATH = "instances_default.json"   # change to your file's path

d = json.load(open(PATH, encoding="utf-8"))
cat = {c["id"]: c["name"] for c in d["categories"]}
anns, imgs = d["annotations"], d["images"]

# Instances per class
n = collections.Counter(cat[a["category_id"]] for a in anns)
print("| Class | Instances |\n|---|---|")
for k in ["head", "thorax", "abdomen", "leg"]:
    print(f"| {k} | {n[k]} |")
print("total instances:", len(anns), "| images:", len(imgs))

# Occluded flag
print("occluded=True:", sum(a.get("attributes", {}).get("occluded", False) for a in anns))

# Flies and recordings, parsed from the file names
def parse(f):
    m = re.search(r"(\d{8})_(Fly\d+).*?(range\d+)", f)
    return m.groups() if m else None

parsed = [parse(i["file_name"]) for i in imgs]
print("unparsed filenames:", parsed.count(None))
parsed = [p for p in parsed if p]
print("dates:", sorted({p[0] for p in parsed}))
print("flies:", len({(p[0], p[1]) for p in parsed}))
print("recordings (fly + range):", len({p for p in parsed}))

# Frames per class check: how many images contain each class
per_img = collections.defaultdict(set)
for a in anns:
    per_img[a["image_id"]].add(cat[a["category_id"]])
print("images with no annotations:", len(imgs) - len(per_img))
for k in ["head", "thorax", "abdomen", "leg"]:
    print(f"images containing {k}:", sum(k in s for s in per_img.values()))