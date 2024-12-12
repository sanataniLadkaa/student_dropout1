import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
from sklearn.preprocessing import LabelEncoder
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost",  # Allow localhost
    "http://localhost:3000",  # Allow frontend app if running on port 3000
    # Add other origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows the listed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Static files (for storing generated charts and prediction results)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create directories if they do not exist
os.makedirs('temp', exist_ok=True)

# Load the trained model and label encoders
model_rf = joblib.load("models/random_forest_model (1).pkl")
label_encoder_qualification = joblib.load("models/label_encoder_qualification.pkl")
label_encoder_area = joblib.load("models/label_encoder_area.pkl")

# Safe Label Encoding
def safe_label_encode(label_encoder, data_column):
    if not hasattr(label_encoder, 'classes_'):
        label_encoder.fit(data_column)

    known_classes = set(label_encoder.classes_)
    data_column = data_column.apply(lambda x: x if x in known_classes else 'Unknown')
    
    if 'Unknown' not in label_encoder.classes_:
        label_encoder.classes_ = np.append(label_encoder.classes_, 'Unknown')

    return label_encoder.transform(data_column)

# Function to preprocess uploaded data
def preprocess_uploaded_data_for_prediction(data):
    data['parents_qualification'] = safe_label_encode(label_encoder_qualification, data['parents_qualification'])
    data['area'] = safe_label_encode(label_encoder_area, data['area'])
    return data

# Function to scale dropout probabilities
def scale_probabilities_to_10(probabilities, threshold=0.5):
    scaled = probabilities.copy()
    scaled[scaled <= threshold] = 0  # Non-dropout, set to 0
    scaled[scaled > threshold] = (scaled[scaled > threshold] - threshold) / (1 - threshold) * 10
    return scaled


def plot_scaled_probability_histogram(data):
    """
    Plot and save a histogram of scaled dropout probabilities.

    Args:
        data (pd.DataFrame): The dataset containing the column 'Scaled_Dropout_Probability'.
        save_path (str): The path to save the generated histogram image.

    Returns:
        str: The file path of the saved histogram.
    """
    # Extract the scaled dropout probabilities
    save_path="static/scaled_probability_histogram.png"
    if 'Scaled_Dropout_Probability' not in data.columns:
        raise ValueError("Dataframe must contain a 'Scaled_Dropout_Probability' column.")
    
    scaled_probabilities = data['Scaled_Dropout_Probability']

    # Create the histogram
    plt.figure(figsize=(8, 5))
    counts, bins, patches = plt.hist(scaled_probabilities, bins=10, color='blue', edgecolor='black', alpha=0.7)

    # Add counts on top of each bar
    for count, bin_edge in zip(counts, bins):
        if count > 0:
            plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, int(count),
                     ha='center', va='bottom', fontsize=10, color='black')

    # Add labels, title, and grid
    plt.title('Histogram of Scaled Dropout Probabilities', fontsize=14)
    plt.xlabel('Scaled Probability (1 to 10)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot to the specified path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    return save_path



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Save uploaded file to a temporary location
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb+") as f:
        f.write(file.file.read())

    try:
        # Read and preprocess the data
        data = pd.read_csv(file_location)
        processed_data = preprocess_uploaded_data_for_prediction(data)
        
        # Make predictions using the trained model
        probabilities = model_rf.predict_proba(processed_data)
        predictions = model_rf.predict(processed_data)
        
        # Scale dropout probabilities to a 1-10 scale
        dropout_probabilities = probabilities[:, 1]
        scaled_probabilities = scale_probabilities_to_10(dropout_probabilities)
        
        # Add predictions and scaled probabilities to the dataframe
        data['Dropout_Prediction'] = predictions
        data['Scaled_Dropout_Probability'] = scaled_probabilities

        # Save the results to a CSV file
        result_filename = f"static/dropout_predictions_scaled.csv"
        data.to_csv(result_filename, index=False)

        # Plot the scaled dropout probabilities
        histogram_path = plot_scaled_probability_histogram(data)

        # Return the results with a flag indicating prediction is done
        return templates.TemplateResponse("results.html", {
            "request": request,
            "prediction_done": True,
            "result_file": result_filename,
            "chart_file": f"/{histogram_path}"  # Correctly point to the histogram
        })

    except Exception as e:
        return {"error": str(e)}



@app.post("/get_student_details", response_class=HTMLResponse)
async def get_student_details_endpoint(
    request: Request,
    student_index: int = Form(None),
    student_id: str = Form(None)
):
    try:
        # Load the results from the previously saved CSV
        data = pd.read_csv("static/dropout_predictions_scaled.csv")
        
        # Retrieve student details
        if student_index is not None:
            student_details = data.iloc[student_index].to_dict()
        elif student_id is not None and 'Student_ID' in data.columns:
            student_details = data[data['Student_ID'] == student_id].to_dict(orient='records')
            student_details = student_details[0] if student_details else None
        else:
            student_details = None

        # If student details are found, pass them to the template
        if student_details:
            return templates.TemplateResponse("student_details.html", {
                "request": request,
                "student_details": student_details,
                "student_found": True
            })
        else:
            return templates.TemplateResponse("student_details.html", {
                "request": request,
                "student_found": False,
                "error_message": "Student not found. Please check the input."
            })

    except Exception as e:
        return {"error": str(e)}
