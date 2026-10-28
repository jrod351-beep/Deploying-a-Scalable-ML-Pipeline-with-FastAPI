
import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


project_path = os.path.dirname(os.path.abspath(__file__))
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
model_path = os.path.join(project_path, "model", "model.pkl")

encoder = load_model(encoder_path)
model = load_model(model_path)

app = FastAPI()


@app.get("/")
async def get_root():
    """Return a welcome message."""
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/data/")
async def post_inference(data: Data):
    """Predict whether a person's income exceeds $50K."""
    data_dict = data.dict(by_alias=True)
    data_dict = {k: [v] for k, v in data_dict.items()}
    data_frame = pd.DataFrame.from_dict(data_dict)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    data_processed, _, _, _ = process_data(
        data_frame,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
    )

    prediction = inference(model, data_processed)
    result = apply_label(prediction)

    return {"result": result}
