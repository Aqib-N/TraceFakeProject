# TraceFake AI System

## Features

- CNN image detection (MobileNetV2)
- EXIF metadata analysis
- Fusion AI prediction system

## Run training

1. python src/preprocess/preprocess_images.py
2. python src/preprocess/exif_extractor.py
3. python src/train/train_exif_model.py
4. python src/train/train_cnn.py

# RUN Ui

streamlit run app_ui.py

./venv/bin/python src/preprocess/preprocess_images.py && \
./venv/bin/python src/preprocess/exif_extractor.py && \
./venv/bin/python src/train/train_exif_model.py && \
./venv/bin/python src/train/train_cnn.py
