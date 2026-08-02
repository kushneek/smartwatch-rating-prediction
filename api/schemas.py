from typing import Optional, List
from pydantic import BaseModel, Field


class SmartwatchInput(BaseModel):
    Brand: str = Field(..., examples=["noise"])
    Current_Price: float = Field(..., alias="Current Price", gt=0)
    Original_Price: float = Field(..., alias="Original Price", gt=0)
    Number_OF_Ratings: float = Field(0, alias="Number OF Ratings", ge=0)
    Dial_Shape: str = Field(..., alias="Dial Shape")
    Strap_Color: str = Field(..., alias="Strap Color")
    Strap_Material: str = Field(..., alias="Strap Material")
    Touchscreen: str = Field(..., examples=["Yes", "No"])
    Battery_Life_Days: float = Field(..., alias="Battery Life (Days)", ge=0)
    Bluetooth: str = Field(..., examples=["Yes", "No"])
    Display_Size: float = Field(..., alias="Display Size", gt=0, le=3)
    Weight: Optional[str] = Field(None, examples=["35 - 50 g"])

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    predicted_rating: float
    model_name: str
    model_cv_r2: float
    warnings: List[str] = []


class ModelInfoResponse(BaseModel):
    model_name: str
    trained_at: str
    cv_r2_mean: float
    cv_r2_std: float
    test_r2: float
    test_mae: float
    n_train_rows: int
    n_features: int