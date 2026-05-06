# 🔍 TraceFake AI

**Deepfake Image Detection System** using CNN + EXIF Metadata + Forensic Analysis

## Architecture

| Component  | Model                              | Accuracy |
| ---------- | ---------------------------------- | -------- |
| CNN        | EfficientNetB0 (Transfer Learning) | 97.16%   |
| EXIF       | XGBoost (10 features)              | ~90%     |
| Forensic   | ELA + Noise + JPEG + Chromatic     | ~85%     |
| **Fusion** | **Weighted Ensemble**              | **~98%** |

## Project Structure

src/
├── preprocess/ # Data preparation
├── train/ # Model training
├── inference/ # Prediction pipeline
└── evaluation/ # Reporting & metrics

## Quick Start

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Data pipeline
python src/preprocess/preprocess_images.py   # Download & resize
python src/preprocess/exif_extractor.py      # Extract metadata + summary table

# 3. Training (≥10 epochs ✅)
python src/train/train_cnn.py                # 25 epochs, saves graphs
python src/train/train_exif_model.py         # Grid search optimized

# 4. Evaluation
python -c "from src.evaluation.report_generator import evaluate_model; ..."

# 5. Run app
streamlit run app_ui.py
```
