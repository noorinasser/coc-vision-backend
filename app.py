import io
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import torch
from transformers import AutoImageProcessor, AutoModel
import os

app = FastAPI()

METADATA_PATH = "/home/nwry/metadata_dinov2.json"

with open(METADATA_PATH, "r", encoding="utf-8") as f:
  METADATA = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
dinov2_model = AutoModel.from_pretrained(model_name).to(device)
dinov2_model.eval()


def extract_dinov2_embedding(pil_image: Image.Image) -> np.ndarray:
  inputs = processor(images=pil_image, return_tensors="pt").to(device)
  with torch.no_grad():
    outputs = dinov2_model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

  embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
  return embedding.astype(np.float32)


def search_top_5_layouts(
    query_embedding: np.ndarray, target_th: Optional[int] = None
):
  scored_items = []
  items_iterable = METADATA.values() if isinstance(METADATA, dict) else METADATA

  for item in items_iterable:
    if not isinstance(item, dict):
      continue

    # تصفية النتائج بناءً على مستوى Town Hall إذا تم تمريره
    if target_th is not None and item.get("th") != target_th:
      continue

    emb_data = item.get("embedding")
    if not emb_data:
      continue

    item_emb = np.array(emb_data, dtype=np.float32)
    sim = np.dot(query_embedding, item_emb) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(item_emb) + 1e-8
    )
    scored_items.append((float(sim), item))

  scored_items.sort(key=lambda x: x[0], reverse=True)
  top_5 = scored_items[:5]

  top_matches = []
  for rank, (sim_score, item) in enumerate(top_5, start=1):
    top_matches.append({
        "rank": rank,
        "similarityScore": round(sim_score, 4),
        "townHallLevel": item.get("th"),
        "baseType": "War Base",
        "officialShareUrl": item.get("link"),
        "deepLinkUrl": item.get("link"),
    })

  return top_matches


@app.post("/api/analyze")
async def analyze_layout(
    file: UploadFile = File(...),
    forced_th: Optional[int] = Form(None),
    api_key: Optional[str] = Form(None),
):
  try:
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
  except Exception as e:
    raise HTTPException(
        status_code=400, detail=f"Invalid image file: {str(e)}"
    )

  query_embedding = extract_dinov2_embedding(pil_image)
  # تمرير forced_th إلى دالة البحث تحت الاسم target_th
  top_matches = search_top_5_layouts(query_embedding, target_th=forced_th)

  effective_th = (
      top_matches[0]["townHallLevel"]
      if (top_matches and top_matches[0]["townHallLevel"])
      else (forced_th if forced_th else 18)
  )

  return JSONResponse(
      content={
          "status": "success",
          "townHallLevel": effective_th,
          "topMatches": top_matches,
      }
  )
