import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import models, transforms

# usage: python predict.py <image_folder> <TeamName> <block>   e.g. demo_test_1 LLMCurators test1A
folder, team, tag = sys.argv[1], sys.argv[2], sys.argv[3]

DEFAULT_CLASSES = ["sp-01", "sp-02", "sp-03", "sp-04", "sp-05"]

obj = torch.load("model.pt", map_location="cpu", weights_only=False)
if isinstance(obj, dict) and "state_dict" in obj:
    state = obj["state_dict"]
    classes = obj.get("class_names", DEFAULT_CLASSES)
else:
    state, classes = obj, DEFAULT_CLASSES

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(state)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

filenames, predictions = [], []
for f in sorted(os.listdir(folder)):
    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = transform(Image.open(os.path.join(folder, f)).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        pred = model(img).argmax(1).item()
    filenames.append(f)               # same loop, so filenames and predictions never drift apart
    predictions.append(classes[pred])
    print(f, "->", classes[pred])

out = f"{team}_{tag}.csv"             # e.g. LLMCurators_test1A.csv
pd.DataFrame({"file": filenames, "predicted": predictions}).to_csv(out, index=False)
print("\nSaved", out, "with", len(filenames), "rows")
