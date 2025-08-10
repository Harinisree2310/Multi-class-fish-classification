🐟 Multiclass Fish Classification 
This project is a complete deep learning pipeline and Streamlit web application that classifies images of fish into different species. It leverages transfer learning to train and evaluate multiple model architectures (like MobileNetV2 and EfficientNetB0), automatically selecting the best-performing one for deployment in an interactive app.FeaturesFlexible Model Training: Scripts to train various powerful, pre-trained architectures or a custom CNN from scratch.Automated Evaluation: A script to automatically evaluate all trained models on a test set and identify the best one based on its F1-score.Efficient Model Selection: Prioritizes lightweight models like MobileNetV2, which are ideal for fast performance without requiring powerful hardware.Interactive Web App: A user-friendly Streamlit interface that allows users to upload a fish image and receive an instant classification with a confidence score.🚀 How to Run This Project1. PrerequisitesPython 3.8+GitA virtual environment manager (like venv)2. SetupFirst, clone the repository and install the required packages into a virtual environment.# Clone the repository from GitHub
git clone [https://github.com/Harinisree2310/Multi-class-fish-classification.git](https://github.com/Harinisree2310/Multi-class-fish-classification.git)

# Navigate into the project directory
cd Multi-class-fish-classification

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows,
use `venv\Scripts\activate`

# Install all required packages
pip install -r requirements.txt

4. Running the Machine Learning
Pipeline
Execute the following scripts from your terminal in order.
# 1. Train the transfer learning models (defined in config.py)
python src/train_transfer_learning.py

# 2. (Optional) Train a custom CNN from scratch
python src/train_cnn.py

# 3. Evaluate all models to find the best one
# This creates the `evaluation_summary.json` and `best_model.h5` files
python src/evaluate_models.py

# 4. Launch the Streamlit web application
streamlit run src/app.py
Your web browser will automatically open with the fish classification app running and ready to use!
