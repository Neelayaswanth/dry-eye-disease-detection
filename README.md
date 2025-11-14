# Dry Eye Disease Detection System

AI-Driven Advanced Techniques for Detecting Dry Eye Disease Using Multi-Source Evidence

## Features

- **User Registration & Login**: Secure user authentication system
- **Dry Eye Prediction**: MLP-based prediction using questionnaire data
- **Eye Disease Prediction**: MobileNet-based image classification
- **Eye Blink Detection**: VGG-19 based blink state detection

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Navigate to the project directory:**
   ```bash
   cd "C:\Users\yaswa\Desktop\Dry Eye Disease\Dry Eye Disease\Dry Eye Disease\Source Code\Eye DIsease"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Method 1: Using Streamlit (Recommended)

```bash
streamlit run Eye.py
```

The application will start and automatically open in your default web browser at:
- **Local URL**: http://localhost:8501
- **Network URL**: (shown in terminal)

### Method 2: Using Python Module

```bash
python -m streamlit run Eye.py
```

### Method 3: Direct Python Execution (Alternative)

```bash
python Eye.py
```

## Project Structure

```
Eye DIsease/
├── Eye.py              # Main entry point (Registration page)
├── Login.py            # Login page
├── Prediction.py       # Main prediction page with all features
├── requirements.txt    # Python dependencies
├── dbs.db             # SQLite database for user data
├── Dataset/           # Training images for eye disease
│   ├── Affected/
│   └── Not/
└── Blink/             # Training images for blink detection
    ├── Closed/
    ├── Open/
    ├── Partial/
    ├── forward_look/
    ├── left_look/
    └── right_look/
```

## Usage

1. **Start the application** using one of the commands above
2. **Register** a new account on the registration page
3. **Login** with your credentials
4. **Select a feature** from the menu:
   - **Dry Eye Prediction**: Upload CSV/Excel file with questionnaire data
   - **Eye Disease Prediction**: Upload an eye image (JPG/PNG)
   - **Eye Blink Detection**: Upload an eye image (JPG/PNG)

## Notes

- **First Run**: The VGG-19 model will train on first use (takes a few minutes). After training, it will be saved and load instantly on subsequent runs.
- **Model Files**: Trained models are saved as `.h5` files in the project directory
- **Database**: User data is stored in `dbs.db` SQLite database

## Troubleshooting

### Port Already in Use
If port 8501 is already in use, Streamlit will automatically use the next available port (8502, 8503, etc.)

### Dependencies Issues
If you encounter import errors, reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### Model Training
- VGG-19 model training happens automatically on first use
- Training progress is shown in the Streamlit interface
- The model is saved after training for faster subsequent runs

## Stopping the Application

Press `Ctrl + C` in the terminal to stop the Streamlit server.

