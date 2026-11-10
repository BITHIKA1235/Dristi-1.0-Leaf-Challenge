import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, torch, torch.nn as nn, pandas as pd
from PIL import Image
from torchvision import models, transforms

folder, team, tag = sys.argv[1], sys.argv[2], sys.argv[3]

ckpt = torch.load("leaf_model.pth", map_location="cpu")
classes = ckpt["class_names"]
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["state_dict"])
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

filenames, predictions = [], []
for f in sorted(os.listdir(folder)):
    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = tf(Image.open(os.path.join(folder, f)).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        pred = model(img).argmax(1).item()
    filenames.append(f)
    predictions.append(classes[pred])

out = f"{team}_prediction_{tag}.csv"
pd.DataFrame({"file": filenames, "predicted": predictions}).to_csv(out, index=False)
print("Wrote", out, "with", len(filenames), "rows")
