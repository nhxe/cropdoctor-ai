import json
from pathlib import Path

train_path = Path(r"C:\Users\manas\PROJECTS\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train")
classes = sorted([p.name for p in train_path.iterdir() if p.is_dir()])

with open("classes.json", "w") as f:
    json.dump(classes, f, indent=2)

print(f"Saved {len(classes)} classes → classes.json")
print(classes[:5])
