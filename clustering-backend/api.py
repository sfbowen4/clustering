import os, glob, torch, uvicorn, cv2, numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import silhouette_score
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from fastapi.staticfiles import StaticFiles
from PIL import Image

app = FastAPI()

app.mount("/static", StaticFiles(directory="images"), name="static")

# Allow CORS from any origin (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper function: auto_kmeans ---
def auto_kmeans(data, k_range=range(2, 11)):
    best_score = -1
    best_k = None
    best_labels = None
    best_centers = None
    for k in k_range:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()
        try:
            score = silhouette_score(data, labels, metric="euclidean")
        except Exception as e:
            score = -1
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_centers = centers
    return best_k, best_centers, best_labels

# --- Global Feature Extractor (ResNet18) ---
class GlobalFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        # Remove the final avgpool and fc layers to extract features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # shape: (B,512)

# --- Request Models ---
class ClusterRequest(BaseModel):
    directory: str
    auto_k: bool = True  # if True, automatically determine optimal k
    num_clusters: int = 4

class LabelRequest(BaseModel):
    labels: dict  # mapping: {cluster_id: label}

class PredictRequest(BaseModel):
    image_path: str

class SegmentRequest(BaseModel):
    image_path: str
    auto_k: bool = True
    k: int = 4

# --- /cluster Endpoint ---
@app.post("/cluster")
def cluster_images(request: ClusterRequest):
    directory = request.directory
    if not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail="Directory not found")
    image_paths = []
    for ext in ('*.jpg', '*.JPEG', '*.png'):
        image_paths.extend(glob.glob(os.path.join(directory, ext)))
    if not image_paths:
        raise HTTPException(status_code=400, detail="No images found")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlobalFeatureExtractor().to(device).eval()
    features_list = []
    valid_paths = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(img_tensor)
            features_list.append(feat.squeeze(0).cpu())
            valid_paths.append(path)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    if not features_list:
        raise HTTPException(status_code=500, detail="Feature extraction failed")
    X = torch.stack(features_list)  # shape: (N, 512)
    data = X.numpy().astype(np.float32)
    
    if request.auto_k:
        best_k, centers, labels = auto_kmeans(data, k_range=range(2, 11))
        k = best_k
    else:
        k = request.num_clusters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()
    
    cluster_results = {i: [] for i in range(k)}
    image_to_cluster = {}
    for path, cid in zip(valid_paths, labels):
        cluster_results[int(cid)].append(path)
        image_to_cluster[path] = int(cid)
    
    return JSONResponse(content={"clusters": cluster_results, "optimal_k": k})

# --- /label_clusters Endpoint ---
@app.post("/label_clusters")
def label_clusters(request: LabelRequest):
    global labels_mapping
    labels_mapping = {int(k): v for k, v in request.labels.items()}
    return {"message": "Labels updated", "labels": labels_mapping}

# --- /train Endpoint ---
@app.post("/train")
def train_classifier():
    global train_directory, image_to_cluster, labels_mapping
    if not image_to_cluster or not labels_mapping:
        raise HTTPException(status_code=400, detail="No training data or labels available")
    image_paths = []
    image_labels = []
    for path, cid in image_to_cluster.items():
        if cid in labels_mapping:
            image_paths.append(path)
            image_labels.append(labels_mapping[cid])
    if not image_paths:
        raise HTTPException(status_code=400, detail="No labeled images found")
    from torch.utils.data import Dataset, DataLoader
    class ImageDataset(Dataset):
        def __init__(self, paths, labels, transform=None, label2idx=None):
            self.paths = paths
            self.labels = labels
            self.transform = transform
            if label2idx is None:
                unique = sorted(set(labels))
                self.label2idx = {label: idx for idx, label in enumerate(unique)}
            else:
                self.label2idx = label2idx
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            label = self.label2idx[self.labels[idx]]
            return img, label
    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    dataset = ImageDataset(image_paths, image_labels, transform=transform_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_classes = len(dataset.label2idx)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 5
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), "trained_cnn.pth")
    return {"message": "Training complete", "model_path": "trained_cnn.pth"}

# --- /predict Endpoint ---
@app.post("/predict")
def predict_image(request: PredictRequest):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not labels_mapping:
        raise HTTPException(status_code=400, detail="No labels available. Train model first.")
    num_classes = len(set(labels_mapping.values()))
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    try:
        model.load_state_dict(torch.load("trained_cnn.pth", map_location=device))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Trained model not found. Train first.")
    model.eval()
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    try:
        img = Image.open(request.image_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error loading image.")
    img_tensor = transform_test(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
    label2idx = {label: idx for idx, label in enumerate(sorted(set(labels_mapping.values())))}
    idx2label = {v: k for k, v in label2idx.items()}
    predicted_label = idx2label[pred.item()]
    return {"predicted_label": predicted_label}

# --- /segment_bbox Endpoint ---
@app.post("/segment_bbox")
def segment_bbox(request: SegmentRequest):
    image_path = request.image_path
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=400, detail="Image not found")
    img = cv2.imread(image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read image")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img_rgb.shape
    pixels = img_rgb.reshape((-1, 3)).astype(np.float32)
    if request.auto_k:
        best_k, centers, labels = auto_kmeans(pixels, k_range=range(2, 11))
        k = best_k
    else:
        k = request.k
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()
    seg_map = labels.reshape((H, W))
    bboxes = {}
    for cluster in range(k):
        ys, xs = np.where(seg_map == cluster)
        if ys.size == 0 or xs.size == 0:
            continue
        bbox = {
            "x_min": int(xs.min()),
            "y_min": int(ys.min()),
            "x_max": int(xs.max()),
            "y_max": int(ys.max())
        }
        bboxes[str(cluster)] = bbox
    return JSONResponse(content={"optimal_k": k, "bounding_boxes": bboxes})

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
