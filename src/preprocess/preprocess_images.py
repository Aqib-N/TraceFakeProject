import random
import hashlib
import piexif
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from datetime import datetime, timedelta

SRC_REAL = Path("/content/drive/MyDrive/real_image_processed")  
SRC_FAKE = Path("/content/drive/MyDrive/fake_image_processed")  

# Output paths
OUT_REAL = Path("data/processed/real")
OUT_FAKE = Path("data/processed/fake")

# Create output directories
OUT_REAL.mkdir(parents=True, exist_ok=True)
OUT_FAKE.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
MAX_IMAGES = 21300

# ==================== MODERN REAL CAMERAS (2020-2025) ====================
REAL_CAMERAS = [
    # Canon
    (b'Canon', b'EOS R5'),
    (b'Canon', b'EOS R6 Mark II'),
    (b'Canon', b'EOS R8'),
    (b'Canon', b'EOS R50'),
    (b'Canon', b'EOS 5D Mark IV'),
    (b'Canon', b'EOS R3'),
    (b'Canon', b'EOS R7'),
    (b'Canon', b'EOS 90D'),
    
    # Nikon
    (b'Nikon', b'Z9'),
    (b'Nikon', b'Z8'),
    (b'Nikon', b'Z7 II'),
    (b'Nikon', b'Z6 II'),
    (b'Nikon', b'Z5'),
    (b'Nikon', b'Zf'),
    (b'Nikon', b'D850'),
    (b'Nikon', b'D780'),
    
    # Sony
    (b'Sony', b'ILCE-1'),  # A1
    (b'Sony', b'ILCE-7RM5'),  # A7R V
    (b'Sony', b'ILCE-7M4'),  # A7 IV
    (b'Sony', b'ILCE-7CM2'),  # A7C II
    (b'Sony', b'ILCE-6700'),  # A6700
    (b'Sony', b'ILCE-9M3'),  # A9 III
    (b'Sony', b'ILCE-7SM3'),  # A7S III
    (b'Sony', b'ZV-E1'),  # Vlogging camera
    
    # Fujifilm
    (b'FUJIFILM', b'X-H2S'),
    (b'FUJIFILM', b'X-H2'),
    (b'FUJIFILM', b'X-T5'),
    (b'FUJIFILM', b'X-S20'),
    (b'FUJIFILM', b'X100VI'),
    (b'FUJIFILM', b'GFX 100 II'),
    (b'FUJIFILM', b'X-T30 II'),
    
    # Panasonic
    (b'Panasonic', b'DC-S5M2'),  # Lumix S5 II
    (b'Panasonic', b'DC-S5M2X'),
    (b'Panasonic', b'DC-G9M2'),  # Lumix G9 II
    (b'Panasonic', b'DC-S1R'),
    (b'Panasonic', b'DC-S5'),
    (b'Panasonic', b'Lumix GH6'),
    
    # Other Real Brands
    (b'Leica', b'M11'),
    (b'Leica', b'Q3'),
    (b'Leica', b'SL3'),
    (b'OM System', b'OM-1 Mark II'),
    (b'OM SYSTEM', b'OM-5'),
    (b'Pentax', b'K-3 Mark III'),
    (b'RICOH', b'GR IIIx'),
    (b'Hasselblad', b'X2D 100C'),
    (b'Hasselblad', b'907X'),
    
    # Modern Smartphones
    (b'Apple', b'iPhone 15 Pro'),
    (b'Apple', b'iPhone 15 Pro Max'),
    (b'Apple', b'iPhone 15'),
    (b'Apple', b'iPhone 14 Pro Max'),
    (b'Samsung', b'Galaxy S24 Ultra'),
    (b'Samsung', b'Galaxy S24+'),
    (b'Samsung', b'Galaxy S23 Ultra'),
    (b'Samsung', b'Galaxy Z Fold 5'),
    (b'Google', b'Pixel 8 Pro'),
    (b'Google', b'Pixel 8'),
    (b'Google', b'Pixel 7 Pro'),
    (b'Xiaomi', b'13 Ultra'),
    (b'Xiaomi', b'14 Pro'),
    (b'OnePlus', b'12'),
    (b'OnePlus', b'Open'),
    (b'Huawei', b'P60 Pro'),
    (b'Huawei', b'Mate 60 Pro'),
    (b'Nothing', b'Phone 2'),
    (b'Vivo', b'X100 Pro'),
    (b'Asus', b'Zenfone 10'),
]

# ==================== MODERN FAKE/AI CAMERAS ====================
FAKE_CAMERAS = [
    # AI Image Generators (Latest)
    (b'Midjourney', b'v6.1'),
    (b'Midjourney', b'v6'),
    (b'Midjourney', b'v5.2'),
    (b'OpenAI', b'DALL-E 3'),
    (b'OpenAI', b'DALL-E 2'),
    (b'Stability AI', b'Stable Diffusion 3.5'),
    (b'Stability AI', b'Stable Diffusion XL'),
    (b'Stability AI', b'Stable Cascade'),
    (b'Google', b'Imagen 2'),
    (b'Meta', b'Imagine'),
    (b'Meta', b'Make-A-Scene'),
    (b'Adobe', b'Firefly 2'),
    (b'Adobe', b'Firefly'),
    (b'Canva', b'AI Generator'),
    (b'Leonardo AI', b'Phoenix'),
    (b'Playground AI', b'v2.5'),
    (b'NightCafe', b'Stable Diffusion'),
    (b'DeviantArt', b'DreamUp'),
    (b'Shutterstock', b'AI'),
    
    # Deepfake Face Swapping
    (b'DeepFaceLab', b'2.0'),
    (b'DeepFaceLab', b'3.0.1'),
    (b'FaceSwap', b'Deepfake'),
    (b'Roop', b'Face Swapper'),
    (b'Roop', b'Roop-Unleashed'),
    (b'InsightFace', b'Swapper'),
    (b'SimSwap', b'Face Swap'),
    (b'FaceSwap', b'GAN'),
    (b'DeepSwap', b'Ai'),
    (b'Reface', b'App'),
    
    # GAN Models
    (b'StyleGAN', b'StyleGAN3'),
    (b'StyleGAN', b'StyleGAN2-ADA'),
    (b'StyleGAN', b'StyleGAN2'),
    (b'ProGAN', b'128x128'),
    (b'BigGAN', b'512x512'),
    (b'StyleGAN-XL', b'Model'),
    (b'GAN', b'Generated'),
    
    # AI Art Generators
    (b'Stable Diffusion', b'DreamShaper'),
    (b'Stable Diffusion', b'Realistic Vision'),
    (b'Midjourney', b'Niji'),
    (b'PixAI', b'Generator'),
    (b'Scenario', b'Gen AI'),
    
    # Suspicious/Unknown
    (b'Unknown', b'AI Generated'),
    (b'Synthetic', b'Deepfake Model'),
    (b'DeepDream', b'AIGenerator'),
    (b'GAN', b'Face Generator'),
    (b'Diffusion', b'Model v2'),
    (b'Neural', b'Face Generator'),
    (b'AI Studio', b'Generator'),
]

# ==================== MODERN REAL SOFTWARE ====================
REAL_SOFTWARE = [
    # Adobe Suite 2024-2025
    b'Adobe Photoshop 2025',
    b'Adobe Photoshop 2024',
    b'Adobe Photoshop 25.0',
    b'Adobe Lightroom Classic 14.0',
    b'Adobe Lightroom Classic 13.0',
    b'Adobe Lightroom 8.0',
    b'Adobe Lightroom 7.0',
    b'Adobe Camera Raw 17.0',
    b'Adobe Camera Raw 16.0',
    b'Adobe Bridge 2025',
    b'Adobe Bridge 2024',
    b'Adobe DNG Converter 16.0',
    
    # Professional Tools
    b'Capture One 23',
    b'Capture One Pro 16.5',
    b'Capture One Pro 16',
    b'DxO PhotoLab 8',
    b'DxO PhotoLab 7',
    b'Luminar Neo 1.20',
    b'Luminar AI',
    b'ON1 Photo RAW 2025',
    b'ON1 Photo RAW 2024',
    b'Affinity Photo 2.5',
    b'Affinity Photo 2',
    b'Pixelmator Pro 3.6',
    b'Darktable 4.8',
    b'Darktable 4.6',
    b'RawTherapee 5.11',
    b'RawTherapee 5.9',
    
    # Open Source
    b'GIMP 2.10.38',
    b'GIMP 3.0',
    b'Krita 5.2',
    b'Krita 5.1',
    
    # Smartphone Apps
    b'Instagram Filter',
    b'Snapseed 2.0',
    b'VSCO',
    b'Lightroom Mobile 9.0',
    b'Snapchat Camera',
    b'FaceTune 2',
    b'Remini',
    
    # Camera Manufacturer Software
    b'Canon Digital Photo Professional 4.18',
    b'Nikon NX Studio 1.6',
    b'Sony Imaging Edge 3.6',
    b'Fujifilm X RAW Studio 2.0',
    b'OM System Workspace',
    b'Panasonic Lumix Tether',
]

# ==================== MODERN FAKE/AI SOFTWARE ====================
FAKE_SOFTWARE = [
    # Deepfake Tools
    b'DeepFaceLab 3.0',
    b'DeepFaceLab 2.0',
    b'FaceSwap Deepfake Pro',
    b'Roop Face Swapper v1.3',
    b'InsightFace Swapper',
    b'SimSwap Face Swap',
    b'DeepSwap AI',
    b'Reface App',
    b'Faceswap GAN',
    b'Deepfake Creator Pro',
    
    # AI Image Generators
    b'Midjourney AI v6.1',
    b'Midjourney AI v6',
    b'DALL-E 3 Generator',
    b'DALL-E 2',
    b'Stable Diffusion 3.5',
    b'Stable Diffusion XL',
    b'Stable Cascade',
    b'Adobe Firefly 2',
    b'Adobe Firefly',
    b'Google Imagen 2',
    b'Meta Imagine',
    b'Leonardo AI Phoenix',
    b'Playground AI v2.5',
    b'NightCafe Creator',
    
    # GAN Tools
    b'StyleGAN3 Generator',
    b'StyleGAN2-ADA',
    b'ProGAN Model',
    b'BigGAN Generator',
    b'GAN Face Generator',
    b'StyleGAN-XL',
    
    # AI Editing
    b'DeepDream AI',
    b'AI Face Editor Pro',
    b'Neural Face Swap',
    b'Deepfake App Pro',
    b'AI Portrait Generator',
    
    # Suspicious
    b'AI Generated Content',
    b'Synthetic Media Creator',
    b'Deepfake Maker Pro',
    b'Face Generator AI',
    b'AI Art Generator',
]

# ==================== SUSPICIOUS KEYWORDS FOR DETECTION ====================
SUSPICIOUS_SOFTWARE_KEYWORDS = [
    'deepfake', 'deepface', 'faceswap', 'roop', 'insightface', 'simswap',
    'midjourney', 'dall-e', 'dalle', 'stable diffusion', 'stability ai',
    'stylegan', 'progan', 'biggan', 'gan', 'diffusion', 'cascade',
    'firefly', 'imagen', 'imagine', 'leonardo', 'playground', 'nightcafe',
    'deepdream', 'neural', 'synthetic', 'generated', 'ai', 'fake',
    'swapper', 'creator', 'generator', 'make', 'model', 'dreamup',
    'reface', 'deepswap'
]

# ==================== FAKE BRANDS (Don't exist) ====================
FAKE_BRANDS = [
    b'AIGenerator', b'DeepfakeAI', b'FakeCamera', b'SyntheticCam',
    b'GAN Cam', b'DiffusionCam', b'StyleGAN Cam', b'NeuralCam',
    b'Midjourney', b'OpenAI', b'Stability AI', b'DeepFaceLab'
]

# ==================== REAL BRANDS (For validation) ====================
REAL_BRANDS = [
    b'Canon', b'Nikon', b'Sony', b'FUJIFILM', b'Panasonic', 
    b'Leica', b'OM SYSTEM', b'OM System', b'Pentax', b'RICOH',
    b'Hasselblad', b'Apple', b'Samsung', b'Google', b'Xiaomi', 
    b'OnePlus', b'Huawei', b'Nothing', b'Vivo', b'Asus'
]

def is_suspicious_software(software_name):
    """Enhanced suspicious software detection"""
    if not software_name:
        return False
    
    name_lower = software_name.lower().decode('utf-8', errors='ignore')
    
    # Check against known suspicious keywords
    for keyword in SUSPICIOUS_SOFTWARE_KEYWORDS:
        if keyword in name_lower:
            return True
    
    return False

def is_valid_camera_brand(make):
    """Check if camera brand is legitimate"""
    if not make:
        return False
    
    make_val = make if isinstance(make, bytes) else make.encode()
    
    # Check against fake brands first
    if make_val in FAKE_BRANDS:
        return False
    
    # Check against real brands
    return make_val in REAL_BRANDS

def generate_realistic_exif(is_fake, has_original_exif=False):
    """Generate realistic EXIF data with modern device patterns"""
    
    if not has_original_exif or random.random() < 0.3:
        # Get current time with realistic offset (last 2 years)
        now = datetime.now()
        random_days = random.randint(-730, 0)
        random_time = now + timedelta(days=random_days)
        timestamp = random_time.strftime("%Y:%m:%d %H:%M:%S").encode()
        
        if is_fake:
            # FAKE images: Use fake cameras, AI software, inconsistent data
            make, model = random.choice(FAKE_CAMERAS)
            software = random.choice(FAKE_SOFTWARE)
            
            # Higher chance of suspicious software for fake images
            suspicious_software = random.random() < 0.8
            
            exif_dict = {
                '0th': {
                    piexif.ImageIFD.Make: make,
                    piexif.ImageIFD.Model: model,
                    piexif.ImageIFD.Software: software,
                    piexif.ImageIFD.DateTime: timestamp,
                    piexif.ImageIFD.Artist: b'AI Generated' if random.random() < 0.8 else b'Unknown',
                    piexif.ImageIFD.Copyright: b'Synthetic Content' if random.random() < 0.7 else b'',
                },
                'Exif': {
                    piexif.ExifIFD.DateTimeOriginal: timestamp,
                    piexif.ExifIFD.DateTimeDigitized: timestamp,
                    # Unrealistic exposure settings
                    piexif.ExifIFD.ExposureTime: (random.choice([1, 10, 100, 500, 1000]), random.choice([50, 100, 500, 1000, 2000])),
                    piexif.ExifIFD.FNumber: (random.choice([1, 1.4, 2, 2.8, 4, 5.6, 8]), 10),
                    piexif.ExifIFD.ISOSpeedRatings: random.choice([50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]),
                    piexif.ExifIFD.FocalLength: (random.choice([18, 24, 35, 50, 85, 135, 200, 300]), 10),
                },
                'GPS': {}  # Fake images typically lack GPS
            }
            
            # Add suspicious software note
            if suspicious_software:
                exif_dict['0th'][piexif.ImageIFD.ImageDescription] = b'Generated by AI Model'
                
        else:
            # REAL images: Use real cameras, professional software, complete metadata
            make, model = random.choice(REAL_CAMERAS)
            software = random.choice(REAL_SOFTWARE)
            
            # Realistic exposure values based on camera type
            if make in [b'Apple', b'Samsung', b'Google', b'Xiaomi', b'OnePlus', b'Huawei']:
                # Smartphones
                exposure = (1, random.choice([50, 60, 100, 120, 200, 500, 1000]))
                fnumber = (random.choice([1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2]), 10)
                iso = random.choice([20, 25, 32, 40, 50, 64, 80, 100, 125, 160, 200])
                focal_length = random.choice([24, 26, 28, 35, 50, 85])
            else:
                # DSLR/Mirrorless
                exposure = (1, random.choice([100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000]))
                fnumber = (random.choice([1.4, 1.8, 2, 2.8, 4, 5.6, 8, 11, 16]), 10)
                iso = random.choice([100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600])
                focal_length = random.choice([24, 35, 50, 70, 85, 105, 135, 200, 300])
            
            exif_dict = {
                '0th': {
                    piexif.ImageIFD.Make: make,
                    piexif.ImageIFD.Model: model,
                    piexif.ImageIFD.Software: software,
                    piexif.ImageIFD.DateTime: timestamp,
                    piexif.ImageIFD.Artist: random.choice([b'Professional Photographer', b'Content Creator', b'Mobile Photographer', b'']),
                    piexif.ImageIFD.Copyright: random.choice([b'All Rights Reserved', b'Creative Commons', b'']),
                    piexif.ImageIFD.XResolution: (72, 1),
                    piexif.ImageIFD.YResolution: (72, 1),
                    piexif.ImageIFD.ResolutionUnit: 2,
                },
                'Exif': {
                    piexif.ExifIFD.DateTimeOriginal: timestamp,
                    piexif.ExifIFD.DateTimeDigitized: timestamp,
                    piexif.ExifIFD.ExposureTime: exposure,
                    piexif.ExifIFD.FNumber: fnumber,
                    piexif.ExifIFD.ISOSpeedRatings: iso,
                    piexif.ExifIFD.FocalLength: (focal_length, 10),
                    piexif.ExifIFD.ExposureProgram: random.choice([1, 2, 3, 4]),  # Manual, Normal, Aperture priority, Shutter priority
                    piexif.ExifIFD.WhiteBalance: random.choice([0, 1]),  # Auto, Manual
                    piexif.ExifIFD.Flash: random.choice([0, 1, 9, 16, 24, 25]),  # Various flash modes
                    piexif.ExifIFD.MeteringMode: random.choice([1, 2, 3, 4, 5, 6]),  # Various metering modes
                },
                'GPS': {
                    piexif.GPSIFD.GPSVersion: b'\x02\x02\x00\x00',
                    piexif.GPSIFD.GPSLatitude: [(random.randint(0, 90), 1), (random.randint(0, 59), 1), (random.randint(0, 59), 1)],
                    piexif.GPSIFD.GPSLongitude: [(random.randint(0, 180), 1), (random.randint(0, 59), 1), (random.randint(0, 59), 1)],
                    piexif.GPSIFD.GPSAltitude: (random.randint(0, 3000), 1) if random.random() < 0.8 else (0, 0),
                    piexif.GPSIFD.GPSDateStamp: timestamp,
                }
            }
            
            # Add processing software for RAW conversions
            if random.random() < 0.5 and make not in [b'Apple', b'Samsung', b'Google']:
                exif_dict['0th'][piexif.ImageIFD.ProcessingSoftware] = b'Adobe Camera Raw'
            
            # Add lens info for interchangeable lens cameras
            if make not in [b'Apple', b'Samsung', b'Google', b'Xiaomi', b'OnePlus', b'Huawei'] and random.random() < 0.7:
                exif_dict['Exif'][piexif.ExifIFD.LensMake] = make
                exif_dict['Exif'][piexif.ExifIFD.LensModel] = random.choice([
                    b'24-70mm f/2.8', b'70-200mm f/2.8', b'50mm f/1.4', b'85mm f/1.8', 
                    b'16-35mm f/2.8', b'24-105mm f/4', b'100-400mm f/5.6'
                ])
        
        return piexif.dump(exif_dict)
    return None

def extract_exif_features(exif_dict):
    """Extract features from EXIF data for training"""
    if not exif_dict:
        return {
            'missing_count': 10,
            'has_camera_info': False,
            'has_software': False,
            'software_suspicious': False,
            'has_timestamp': False,
            'timestamp_consistent': False,
            'exif_total_tags': 0,
            'has_gps': False,
            'has_flash': False,
            'has_orientation': False
        }
    
    try:
        exif = piexif.load(exif_dict)
        
        has_camera = piexif.ImageIFD.Make in exif['0th'] and piexif.ImageIFD.Model in exif['0th']
        has_software = piexif.ImageIFD.Software in exif['0th']
        
        software_value = exif['0th'].get(piexif.ImageIFD.Software, b'').decode('utf-8', errors='ignore')
        suspicious_software = is_suspicious_software(software_value)
        
        has_timestamp = piexif.ExifIFD.DateTimeOriginal in exif['Exif']
        has_gps = bool(exif['GPS'])
        has_flash = piexif.ExifIFD.Flash in exif['Exif']
        
        total_tags = sum(len(v) for v in exif.values())
        
        return {
            'missing_count': max(0, 20 - total_tags),  # Fewer tags = more suspicious
            'has_camera_info': 1 if has_camera else 0,
            'has_software': 1 if has_software else 0,
            'software_suspicious': 1 if suspicious_software else 0,
            'has_timestamp': 1 if has_timestamp else 0,
            'timestamp_consistent': 1,  # Can be enhanced with consistency checks
            'exif_total_tags': total_tags,
            'has_gps': 1 if has_gps else 0,
            'has_flash': 1 if has_flash else 0,
            'has_orientation': 1 if piexif.ImageIFD.Orientation in exif['0th'] else 0
        }
    except:
        return {
            'missing_count': 10,
            'has_camera_info': False,
            'has_software': False,
            'software_suspicious': False,
            'has_timestamp': False,
            'timestamp_consistent': False,
            'exif_total_tags': 0,
            'has_gps': False,
            'has_flash': False,
            'has_orientation': False
        }

# Fast safe hash function
def get_hash(fp):
    try:
        h = hashlib.md5()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()
    except:
        return None

# Image processing function - PRESERVES AND ADDS metadata
def process(fp, out_dir, label, idx, is_fake):
    try:
        # Open image
        img = Image.open(fp)
        
        # Extract original EXIF data
        original_exif = img.info.get('exif', b'')
        has_original_exif = len(original_exif) > 0
        
        # Generate or enhance EXIF data
        enhanced_exif = generate_realistic_exif(is_fake, has_original_exif)
        
        # Extract features for CSV export
        features = extract_exif_features(enhanced_exif if enhanced_exif else original_exif)
        features['image_path'] = f"{label}_{idx}.jpg"
        features['label'] = 1 if is_fake else 0  # 1=fake, 0=real
        
        # Convert to RGB if needed
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        
        # Save with metadata
        save_kwargs = {
            'format': 'JPEG',
            'quality': 95,
            'optimize': False,
            'progressive': False
        }
        
        # Use enhanced EXIF if generated, otherwise keep original
        if enhanced_exif:
            save_kwargs['exif'] = enhanced_exif
        elif original_exif:
            save_kwargs['exif'] = original_exif
        
        img.save(out_dir / f"{label}_{idx}.jpg", **save_kwargs)
        return features
    except UnidentifiedImageError:
        return None
    except Exception as e:
        print(f"Error processing {fp}: {e}")
        return None

# Preprocessing function with CSV export
def preprocess(src, dst, label, is_fake):
    if not src.exists():
        print(f"⚠️ Warning: Source path does not exist: {src}")
        print(f"Please update SRC_REAL and SRC_FAKE paths to your Google Drive directories")
        return []
    
    print(f"\nScanning {src} for images...")
    files = list(src.rglob("*"))
    random.shuffle(files)

    seen_hashes = set()   
    count = 0
    skipped_dup = 0
    skipped_bad = 0
    features_list = []

    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    for fp in files:
        if count >= MAX_IMAGES:
            break

        if fp.suffix.lower() not in valid_ext:
            continue

        h = get_hash(fp)
        if h is None:
            skipped_bad += 1
            continue

        if h in seen_hashes:
            skipped_dup += 1
            continue

        seen_hashes.add(h)

        result = process(fp, dst, label, count, is_fake)
        if result:
            features_list.append(result)
            count += 1

        # Progress indicator
        if count % 500 == 0 and count > 0:
            print(f"Processed {count}/{MAX_IMAGES} {label} images...")

    print(f"""
========================
{label.upper()} DONE
Processed: {count}
Duplicates skipped: {skipped_dup}
Bad files skipped: {skipped_bad}
EXIF metadata: ADDED/ENHANCED ✓
========================
""")
    return features_list

if __name__ == "__main__":
    import pandas as pd
    
    print("=" * 50)
    print("Processing REAL images from Google Drive...")
    print("Adding realistic EXIF metadata to images without it")
    print("=" * 50)
    real_features = preprocess(SRC_REAL, OUT_REAL, "real", is_fake=False)
    
    print("\n" + "=" * 50)
    print("Processing FAKE images from Google Drive...")
    print("Adding suspicious EXIF metadata to fake images")
    print("=" * 50)
    fake_features = preprocess(SRC_FAKE, OUT_FAKE, "fake", is_fake=True)
    
    # Save EXIF features to CSV for training
    all_features = real_features + fake_features
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv("data/exif_features.csv", index=False)
        print(f"\n✅ EXIF features saved to data/exif_features.csv")
        print(f"   Total samples: {len(df)}")
        print(f"   Features: {', '.join([c for c in df.columns if c not in ['image_path', 'label']])}")
    
    print("\n" + "=" * 50)
    print("✅ COMPLETE! Processed images saved to:")
    print(f"   Real images: {OUT_REAL.absolute()}")
    print(f"   Fake images: {OUT_FAKE.absolute()}")
    print(f"   EXIF features CSV: data/exif_features.csv")
    print("=" * 50)