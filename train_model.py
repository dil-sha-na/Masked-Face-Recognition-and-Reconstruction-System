import cv2
import os
import pickle
import numpy as np
import face_recognition

# --- CONFIGURATION ---
DATASET_DIR = "dataset_unmasked"
DB_FILE = "features_db.pkl"
OUTPUT_UPPER = "db_upper"
OUTPUT_LOWER = "db_lower"

os.makedirs(OUTPUT_UPPER, exist_ok=True)
os.makedirs(OUTPUT_LOWER, exist_ok=True)

def apply_clahe(rgb_img):
    """Applies Contrast Limited Adaptive Histogram Equalization to normalize lighting."""
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def train_dataset():
    database = {}  
    print("Starting Phase 1: Dataset Preparation...")
    
    for file in os.listdir(DATASET_DIR):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        path = os.path.join(DATASET_DIR, file)
        name = file.split('_')[0]
        print(f"Processing file: {file}")
        
        try:
            img = cv2.imread(path)
            if img is None:
                continue
                
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_img = apply_clahe(rgb_img)
            
            face_locations = face_recognition.face_locations(rgb_img)
            if not face_locations:
                print(f"  > No face found in {file}, skipping.")
                continue
                
            top, right, bottom, left = face_locations[0]
            
            # Save the lower half for reconstruction purposes
            h = bottom - top
            w = right - left
            split_y = int(h * 0.55)
            
            lower_part = img[top+split_y:bottom, left:right]
            lower_path = os.path.join(OUTPUT_LOWER, f"{name}.jpg")
            cv2.imwrite(lower_path, lower_part)
            
            # FIX: No more blackout! Extract embedding from the natural face bounding box
            encodings = face_recognition.face_encodings(rgb_img, known_face_locations=[(top, right, bottom, left)])
            
            if encodings:
                if name not in database:
                    database[name] = {
                        "encodings": [],
                        "lower_path": lower_path
                    }
                database[name]["encodings"].append(encodings[0])
                print(f"  > Successfully processed: {name} (Total encodings: {len(database[name]['encodings'])})")
            else:
                print(f"  > Failed to generate encoding for {name}")
                
        except Exception as e:
            print(f"Error processing {file}: {e}")

    with open(DB_FILE, 'wb') as f:
        pickle.dump(database, f)
    
    print(f"Training Complete. Database saved to {DB_FILE} with {len(database)} identities.")
    return True

if __name__ == "__main__":
    train_dataset()