# ChestAI
An AI-powered full-stack system for large-scale chest X-ray triage using Spark, PyTorch, and FastAPI

Data you can use (image, text, or both)

Chest X-ray images (image-only or image+labels)

NIH ChestX-ray14 — ~112k de-identified CXRs with 14 pathology labels. Downloadable via NIH or Google Cloud mirrored link. Great for multi-label classification (atelectasis, effusion, etc.). 
Google Cloud
+2
PMC
+2

CheXpert (Stanford) — 224k CXRs with uncertainty-aware labels; includes associated radiology reports, so it’s perfect for multimodal (image+text) projects or report-aware training. Access requires a simple application. Also see CheXpert+ (paired text–image). 
stanfordmlgroup.github.io
+2
AI in Medicine & Imaging Center
+2

MIMIC-CXR (PhysioNet) — 371k images, de-identified, with structured labels and companion datasets (embeddings, biomarkers). Credentialed access with DUA + CITI training—great for a serious, “industry-grade” portfolio piece. 
PhysioNet
+2
PhysioNet
+2

RSNA Pneumonia (Kaggle) — labeled bounding boxes for pneumonia (detection/localization). Ideal if you want to demo object detection (Faster R-CNN/YOLO) and an interactive viewer. 
Kaggle
+2
Kaggle
+2

Clinical notes / reports (text)

n2c2 / i2b2 clinical notes — de-identified discharge summaries and annotated corpora for concept extraction, assertions, etc. Requires registration + Data Use Agreement; gold standard for clinical NLP. 
N2C2
+2
i2b2
+2

Three punchy project tracks (pick one)

Image triage (multi-label)

Data: NIH or CheXpert.

Task: predict pathologies + triage score (“urgent vs routine”).

Spark: bulk image I/O → resize/normalize → write Parquet metadata; compute dataset stats, class imbalance, and shard TFRecord/Parquet for training at scale.

AI: CNN/ViT (or start from public CXR embeddings from PhysioNet to speed things up). 
PhysioNet

Report-aware triage (multimodal)

Data: CheXpert (images + reports) or MIMIC-CXR with reports.

Task: joint image+text model to predict “needs radiologist now,” generate explanatory snippets from the report, or do vision–language retrieval (find similar prior cases).

Spark: text preprocessing at scale (tokenization, section splitting), join on study IDs, build training triplets.

Detection & workflow

Data: RSNA Pneumonia.

Task: object detection for suspected opacities → surface regions with bounding boxes to a reviewer UI; store reviewer feedback to close the loop.

Spark: batch DICOM→PNG conversion logs, label joins, and evaluation (mAP) across folds. 
Kaggle

Architecture (resume-friendly)

Ingestion/ETL: Spark (PySpark) jobs to parse DICOM/PNG, extract metadata, balance classes, and write Delta/Parquet.

Warehouse: PostgreSQL for app state + study metadata; S3/MinIO for images; optional Elasticsearch for case search.

Training: PyTorch/Lightning; optionally use Spark to distribute feature prep and to launch parallel hyperparams.

API: FastAPI inference service (CPU/GPU) returning labels, scores, and saliency maps.

Frontend: React + viewer (CornerstoneJS) for CXRs; triage queue, filters, and “send to review” workflow.

MLOps: MLflow for runs + model registry; CI to retrain nightly on new shards.

What to put on the README (to impress)

Dataset(s) + sizes you used and the access pathway (e.g., “MIMIC-CXR via PhysioNet, DUA completed”). 
PhysioNet

Diagram: ingestion → Spark feature store → training → inference service → web UI.

Metrics: AUROC/F1; latency; throughput (“preprocessed 300k images with Spark in X minutes”).

Ethics: de-identification, DUA compliance, dataset bias notes (domain shift across NIH/CheXpert/MIMIC). 
PhysioNet

If you tell me which track you want (image-only, multimodal, or detection), I’ll hand you a step-by-step build plan (repo layout, Spark scripts, model baselines, and a seed Postgres schema).

You said:
I want to go with the first project track. Go super in-detail on the whole project and how it would be implemented. Mention every tech I would use, what AI algorithm I would use, etc. Give me all the information I need as in-detail as possible to begin implementing
ChatGPT said:
Thought for 1m 23s
Project Blueprint: AI Triage for Chest X-rays (Multi-label)

Below is an end-to-end, implementation-ready plan for the image-only triage track. It covers: data, storage, Spark ETL, model & training, explainability, backend APIs, frontend viewer, database schema, deployment, MLOps, testing, and metrics—with concrete tech choices and code skeletons so you can start building today.

0) TL;DR Stack

Data: NIH ChestX-ray14 (frontal CXRs, 14 labels). Download via NIH/Box or the GCP public mirror. 
PMC
+1

Compute: Python 3.11, PyTorch (GPU), PySpark on Spark 4.x, MLflow for tracking/registry. 
Apache Spark
+1

Storage: S3/MinIO (raw & processed), Delta Lake tables for curated metadata, PostgreSQL for app state. 
Databricks Documentation

Backend: FastAPI + Uvicorn/Gunicorn; optional Celery + Redis for batch jobs. 
FastAPI

Frontend: React + TypeScript (Vite). Image viewer (PNG overlay) now; CornerstoneJS if/when you add DICOM. 
cornerstonejs.org
+1

Model: DenseNet-121 (ImageNet-pretrained), 14-head sigmoid, BCEWithLogitsLoss with per-class pos_weight, input 320–512 px.

Explainability: Grad-CAM overlays via pytorch-grad-cam. 
PyPI
+1

1) Data
1.1 Source & labels

NIH ChestX-ray14: ~112,120 frontal CXRs from ~30k patients; each image has 0–14 labels (Atelectasis, Effusion, Pneumonia, etc.). Labels were mined from reports, so treat as weak supervision. 
PMC

Download from NIH (Box) or the GCP public mirror (PNG). The GCP mirror lists “100,000” images (a widely used subset/mirror), but the canonical paper + many studies reference 112,120. Use whichever you can fetch reliably and keep your README explicit about counts. 
Google Cloud
+1

1.2 Files you’ll have

/raw/NIH/images/ (PNG files)

/raw/NIH/Data_Entry_2017.csv → image name, findings (pipe-delimited)

Optional patient metadata CSVs (age/sex) depending on mirror

1.3 Class selection

For triage, define a subset of urgent-leaning labels you’ll emphasize in a triage score: e.g., Edema, Consolidation, Effusion, Pneumonia, Pneumothorax, Cardiomegaly. Keep all 14 for supervision; use a weighted score for triage.

2) Storage & Layout

Object store: S3 (or MinIO for local dev).

Delta Lake curated tables (Spark-managed) for metadata splits & stats:

delta://cxr/meta_images (image_id, path, height, width, modality, split)

delta://cxr/labels (image_id, multi-hot 14-vector, label_names array)

delta://cxr/stats (class prevalence, pos_weight vector, per-split counts)

PostgreSQL (app DB): users, studies, predictions, reviews, audit.

Directory draft:

s3://cxr-lake/
  raw/nih/{images,labels}
  bronze/ ... (staged ingests)
  silver/ ... (cleaned joins)
  gold/   ... (analysis-ready Delta tables)
  models/
  artifacts/


Delta Lake gives you ACID & schema evolution on the lake. 
Databricks Documentation

3) Spark ETL (PySpark)

Goal: build clean, patient-wise splits; compute class stats & pos_weight; write Delta tables to drive training & evaluation.

Ingest

# spark-submit etl_ingest.py
df_labels = spark.read.csv("s3://cxr-lake/raw/nih/Data_Entry_2017.csv", header=True)
df_imgs = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.png") \
  .load("s3://cxr-lake/raw/nih/images/")


Normalize labels (split pipe-delimited list → 14-dim multi-hot).

Patient-wise split (e.g., 80/10/10 by unique patient ID to prevent leakage). The literature uses patient-wise splits—note it in your README. 
PMC

Image metadata (height/width via a UDF or parse from PNG header).

Class stats → compute prevalence per label & the pos_weight vector used by BCEWithLogitsLoss:

# pos_weight = (N_neg / N_pos) per class
# Store in delta://cxr/stats as an array<double>


Write Delta:

df_meta.write.format("delta").mode("overwrite").save("s3://cxr-lake/gold/cxr/meta_images")
df_labels.write.format("delta").mode("overwrite").save("s3://cxr-lake/gold/cxr/labels")
df_stats.write.format("delta").mode("overwrite").save("s3://cxr-lake/gold/cxr/stats")


Spark docs (4.x) for reference. 
Apache Spark

Why not store pixel tensors in Spark? Keep heavy pixels in object storage and load them directly in PyTorch for training; use Spark to organize metadata, splits, and statistics.

4) Model & Training
4.1 Architecture

Backbone: DenseNet-121 (ImageNet-pretrained).

Head: Linear layer → 14 logits → sigmoid for multi-label.

Input: 1-channel grayscale replicated to 3 channels; 320–512 px shorter side (512 if GPU VRAM allows).

4.2 Loss & imbalance

BCEWithLogitsLoss(pos_weight=...), where pos_weight is the Spark-computed vector. This is the most robust first step for heavy class imbalance.

4.3 Augmentations

Light geometric/photometric: Random resize-crop (±10%), horizontal flip = False by default (laterality matters); small rotations (±5°), random brightness/contrast/CLAHE; center crop for eval.

Normalization: mean/std of ImageNet or dataset-computed.

4.4 Optimization

AdamW, LR 1e-4 (cosine decay), weight_decay 1e-4, epochs 20–30, batch 16–32 (FP16 mixed precision), gradient clip 1.0, early stopping on macro AUROC.

4.5 Evaluation

Per-class AUROC & AUPRC + macro & micro averages.

Thresholds: pick per-class thresholds maximizing Youden’s J or F1 on val set.

Patient-wise evaluation (respect your split).

4.6 Training skeleton (PyTorch Lightning flavored)
class CXRModule(pl.LightningModule):
    def __init__(self, pos_weight):
        super().__init__()
        self.backbone = torchvision.models.densenet121(weights="IMAGENET1K_V1")
        in_feats = self.backbone.classifier.in_features
        self.backbone.classifier = torch.nn.Linear(in_feats, 14)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def training_step(self, batch, _):
        x, y = batch["image"], batch["label"]
        logits = self.backbone(x)
        loss = self.criterion(logits, y)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        return {"optimizer": opt, "lr_scheduler": sch}

4.7 DataLoader

A Dataset that reads the path from the Delta table (exported to a Parquet/CSV manifest per split) and loads PNG with Pillow/OpenCV, applies augmentations (Albumentations or TorchVision transforms).

5) Explainability (Grad-CAM)

Use pytorch-grad-cam to generate class-specific heatmaps (e.g., for “Effusion”). Overlay onto the grayscale CXR for the clinician view. 
PyPI
+1

Store heatmaps alongside predictions in S3; the API can return a signed URL or base64 PNG.

6) Triage Score

Compute a scalar triage score to prioritize review:

triage = max(
  w_edema * p[Edema],
  w_pneumonia * p[Pneumonia],
  w_effusion * p[Effusion],
  w_pneumothorax * p[Pneumothorax],
  w_consolidation * p[Consolidation],
  w_cardiomegaly * p[Cardiomegaly]
)


Start with all w = 1.0, tune later based on validation set clinical priority.

Bucket: Urgent (≥0.8), Soon (0.5–0.8), Routine (<0.5).

7) Backend APIs (FastAPI)

Why FastAPI: type-hinted, production-ready, auto-docs (OpenAPI/Swagger) at /docs. 
FastAPI

Endpoints:

POST /infer — body: image file or S3 path → returns {per_class_probs, triage, thresholds_used}

POST /batch_infer — S3 prefix + optional filters → async job id

GET /jobs/{id} — job status & results

GET /images/{image_id}/heatmap?label=Effusion — returns overlay PNG

Serving:

Uvicorn in dev; Gunicorn+Uvicorn workers in prod; CUDA container if on GPU.

Celery + Redis for batch workloads.

Simple route skeleton:

@app.post("/infer")
def infer(file: UploadFile):
    img = PIL.Image.open(file.file).convert("L")
    x = preprocess(img)  # to tensor (3ch)
    with torch.inference_mode(), torch.cuda.amp.autocast():
        logits = model(x.unsqueeze(0).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    triage = compute_triage(probs)
    return {"probs": dict(zip(LABELS, probs.tolist())), "triage": triage}


Model registry: load the current “Production” model from MLflow Model Registry on startup. 
MLflow

8) Frontend (React + TS)

Pages:

Queue: filters (split, triage bucket, label), table of studies

Viewer: image + Grad-CAM overlay toggle, per-class probabilities, thresholds

Review form: mark as correct/incorrect, comments

Image rendering:

NIH PNGs → just <img> + Canvas overlay (heatmap).

Add CornerstoneJS later if you ingest DICOM (supports DICOMweb, GPU-accelerated). 
cornerstonejs.org

State: TanStack Query (server state); Zustand/Redux if needed.

9) Database Schema (PostgreSQL)
CREATE TABLE users (
  id UUID PRIMARY KEY, email TEXT UNIQUE NOT NULL, role TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE images (
  image_id TEXT PRIMARY KEY,
  patient_id TEXT,
  s3_path TEXT NOT NULL,
  split TEXT CHECK (split IN ('train','val','test')),
  height INT, width INT
);

CREATE TABLE labels (
  image_id TEXT REFERENCES images(image_id),
  label_name TEXT,  -- one row per active label OR store as JSONB for multi-hot
  is_positive BOOLEAN,
  PRIMARY KEY (image_id, label_name)
);

CREATE TABLE predictions (
  id BIGSERIAL PRIMARY KEY,
  image_id TEXT REFERENCES images(image_id),
  model_version TEXT NOT NULL,
  probs JSONB NOT NULL,        -- {label: prob}
  triage_score DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE reviews (
  id BIGSERIAL PRIMARY KEY,
  image_id TEXT REFERENCES images(image_id),
  reviewer UUID REFERENCES users(id),
  label_name TEXT,
  verdict TEXT CHECK (verdict IN ('TP','FP','TN','FN')),
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

10) MLOps

Tracking & Registry: MLflow for metrics, artifacts, and model versioning (Staging → Production). 
MLflow

Data versioning: tag Delta tables by commit/version; record dataset snapshot ID in MLflow run params.

CI/CD: GitHub Actions

Lint/type-check (ruff, mypy), unit tests (pytest), build & push Docker images, run smoke tests against staging API, then promote model in MLflow.

Infra:

Local dev: docker-compose (MinIO, Postgres, API, UI).

Cloud: ECS/EKS + S3 + RDS; or a single GPU EC2 with dockerized services.

11) Security & Compliance

NIH data is de-identified; still treat everything like PHI: HTTPS only, S3 server-side encryption, least-privilege IAM, signed URLs, audit trails.

If you later use CheXpert or MIMIC-CXR (DUA + training required), document compliance steps explicitly in the README. 
stanfordmlgroup.github.io

12) Metrics & Baselines (what “good” looks like)

Expect macro AUROC in the 0.80–0.90 range on common labels at 320–512 px with DenseNet-121 after 20–30 epochs.

Report: per-class AUROC/AUPRC, confusion at chosen thresholds, latency (ms/image), throughput (img/s), and calibration (reliability curves).

13) Repo Layout
cxr-triage/
  README.md
  docker/           # Dockerfiles (api, trainer, spark)
  infra/            # IaC templates (optional)
  api/
    app.py          # FastAPI
    routers/
    models/         # pydantic
  ui/               # React + TS + Vite
  data/
    manifests/      # train/val/test parquet/csv lists
  spark/
    etl_ingest.py
    etl_stats.py
  training/
    datamodule.py
    model.py
    train.py
    eval.py
  mlflow/
  scripts/

14) Build Plan (4 sprints)

Sprint 1 – Data & ETL

Fetch NIH images + labels; write Spark jobs to: ingest, normalize labels, patient-wise split, compute class stats, write Delta tables. (Spark docs / Delta docs). 
Apache Spark
+1

Sprint 2 – Baseline Model

DenseNet-121 baseline with BCEWithLogitsLoss(pos_weight); log runs to MLflow; export “best” to registry. 
MLflow

Sprint 3 – API & UI

FastAPI /infer; React viewer with probability table + heatmap overlay; wire to Postgres. 
FastAPI

Add Grad-CAM overlays and store with predictions. 
PyPI

Sprint 4 – Hardening

Batch inference (Celery), auth (JWT), CI/CD, alerting, dashboards, README with data ethics & limitations.

15) Code Nuggets to Get You Moving
15.1 Export split manifests from Delta (for PyTorch)
spark.read.format("delta").load("s3://.../gold/cxr/meta_images").createOrReplaceTempView("meta")
spark.read.format("delta").load("s3://.../gold/cxr/labels").createOrReplaceTempView("labs")

manifest = spark.sql("""
SELECT m.image_id, m.path, m.split,
       /* 14 binary columns */ 
       CASE WHEN array_contains(labs, 'Atelectasis') THEN 1 ELSE 0 END as Atelectasis,
       ...
FROM meta m
JOIN (
  SELECT image_id, collect_set(label_name) as labs
  FROM labels WHERE is_positive = true GROUP BY image_id
) l ON l.image_id = m.image_id
""")
manifest.write.mode("overwrite").parquet("s3://.../gold/cxr/manifests/")

15.2 PyTorch Dataset
class CXRDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_df, transforms):
        self.df = manifest_df
        self.transforms = transforms
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(fs.open(row.path, 'rb')).convert('L')  # 1ch
        img = img.resize((512,512))
        img = np.repeat(np.array(img)[...,None], 3, axis=-1)
        x = self.transforms(Image.fromarray(img))
        y = torch.tensor(row[LABELS].values.astype(np.float32))
        return {"image": x, "label": y}

15.3 Grad-CAM overlay
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layer = model.backbone.features[-1]
cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=True)
grayscale_cam = cam(input_tensor=x.unsqueeze(0), targets=[ClassifierOutputTarget(idx)])
overlay = show_cam_on_image(norm_img, grayscale_cam[0], use_rgb=True)


(See pytorch-grad-cam docs for details.) 
PyPI

16) Documentation & Citations in README

Link dataset(s) + counts used; note that NIH counts vary by mirror (100k vs 112k); specify your exact totals. 
Google Cloud
+1

Cite Spark, Delta, FastAPI, MLflow, Grad-CAM, CornerstoneJS where applicable. 
cornerstonejs.org
+5
Apache Spark
+5
Databricks Documentation
+5

17) Stretch Goals (high-impact polish)

Calibration (temperature scaling / isotonic regression).

Ensembling or test-time augmentation to squeeze more AUROC.

Active learning loop using reviewer feedback.

CheXpert-trained model variant and compare uncertainty handling (U-ignore vs U-as-pos). 
stanfordmlgroup.github.io
+1

CornerstoneJS DICOM viewer for hospital-grade UX. 
cornerstonejs.org
