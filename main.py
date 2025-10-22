from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware

from transformers import pipeline
from request_types import REQUEST_TYPES

classifier = pipeline("text-classification", model="./airline_classifier_model", tokenizer="./airline_classifier_model")

def label_to_request_type(label: str) -> str:
    # Simple mapping based on similarity
    max_matches = 0
    predicted_type = ""  # Default type

    for req_type in REQUEST_TYPES:
        matches = sum(1 for word in req_type.split('_') if word in label.lower())
        if matches > max_matches:
            max_matches = matches
            predicted_type = req_type

    return predicted_type

app = FastAPI()
app.add_middleware( CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"], )

# feedback_storage = []
# metrics = {"correct": 0, "incorrect": 0}

class ClassifyRequest(BaseModel):
    message: str

# class FeedbackRequest(BaseModel):
#     message: str
#     predicted_type: str
#     feedback: str  # "correct" or "incorrect"

def classify_message(message: str) -> str:
    result = classifier(message)
    label = result[0]['label']
    return label_to_request_type(label)

@app.post("/classify")
async def classify(req: ClassifyRequest):
    req_type = classify_message(req.message)
    return {"request_type": req_type}

# @app.post("/feedback")
# async def feedback(req: FeedbackRequest):
#     feedback_storage.append(req.dict())
    # if req.feedback == "correct":
#         metrics["correct"] += 1
#     elif req.feedback == "incorrect":
#         metrics["incorrect"] += 1
#     return {"message": "Feedback received", "metrics": metrics}

# @app.get("/metrics")
# async def get_metrics():
#     total = metrics["correct"] + metrics["incorrect"]
#     accuracy = metrics["correct"] / total if total else None
#     return {"metrics": metrics, "accuracy": accuracy}
