import cv2
import pickle
import numpy as np
import os
import face_recognition
import time

class VideoCamera(object):
    def __init__(self):
        try:
            self.video = None
            for idx in [0, 1, 2]:
                self.video = cv2.VideoCapture(idx)
                if self.video.isOpened():
                    print(f"Opened camera index {idx}")
                    self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    break
            if not self.video.isOpened():
                raise ValueError("Could not open video device")
        except Exception as e:
            print(f"Error opening camera: {e}")
            self.video = None
        
        # --- CONFIGURATION ---
        self.DB_FILE = "features_db.pkl"
        self.ENCODING_THRESHOLD = 0.55  # Point 4: Strict Euclidean distance threshold (lower is stricter)
        self.MIN_ROI_SIZE = (100, 100)
        self.DNN_CONFIDENCE_THRESHOLD = 0.7
        self.TRACK_THRESHOLD = 3 
        
        # Fallbacks
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        try:
            self.face_net = cv2.dnn.readNetFromCaffe(
                'deploy.prototxt',
                'res10_300x300_ssd_iter_140000.caffemodel'
            )
            self.use_dnn = True
        except Exception as e:
            print(f"Could not load DNN detector. Falling back to Haar cascades")
            self.use_dnn = False
        
        self.load_database()
        self.frame_count = 0
        self.face_tracker = {} 

    def load_database(self):
        if os.path.exists(self.DB_FILE):
            with open(self.DB_FILE, 'rb') as f:
                self.database = pickle.load(f)
            print("Database loaded successfully.")
        else:
            self.database = {}
            print("Warning: Database not found. Please train the model.")

    def reload_database(self):
        self.load_database()
        self.face_tracker = {}

    def release(self):
        if self.video and self.video.isOpened():
            self.video.release()
        self.video = None

    def __del__(self):
        self.release()

    def apply_clahe(self, rgb_img):
        """Point 7: Contrast equalization for robustness against lighting variations"""
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    def detect_faces_dnn(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.DNN_CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                x, y = max(0, x), max(0, y)
                x2, y2 = min(w, x2), min(h, y2)
                faces.append((x, y, x2-x, y2-y))
        return faces

    def validate_face_region(self, frame, x, y, w, h):
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]: return False, "Out of bounds"
        roi = frame[y:y+h, x:x+w]
        if roi.shape[0] < 50 or roi.shape[1] < 50: return False, "Too small"
        aspect_ratio = w / h
        if not (0.5 <= aspect_ratio <= 1.6): return False, "Bad aspect ratio"
        return True, "Valid"

    def assess_face_quality(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50: return False, "Very Blurry", 0
        brightness = np.mean(gray)
        if brightness < 25 or brightness > 235: return False, "Extreme lighting", 0
        if roi.shape[0] < 60 or roi.shape[1] < 60: return False, "Too small", 0
        return True, "Good", 100

    def _iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = (w1 * h1) + (w2 * h2) - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def track_faces(self, current_faces):
        validated_faces = []
        matched_ids = set()
        
        for (x, y, w, h) in current_faces:
            matched = False
            best_iou = 0
            best_id = None
            
            for face_id, data in self.face_tracker.items():
                if face_id in matched_ids: continue
                iou = self._iou((x, y, w, h), data['bbox'])
                if iou > 0.5 and iou > best_iou:
                    best_iou, best_id, matched = iou, face_id, True
            
            if matched and best_id:
                self.face_tracker[best_id]['frames'] += 1
                self.face_tracker[best_id]['bbox'] = (x, y, w, h)
                matched_ids.add(best_id)
                if self.face_tracker[best_id]['frames'] >= self.TRACK_THRESHOLD:
                    validated_faces.append((x, y, w, h))
            else:
                face_id = f"face_{time.time()}"
                self.face_tracker[face_id] = {'bbox': (x, y, w, h), 'frames': 1}
        
        # --- FIXED SECTION ---
        # Replaced the invalid walrus operator with a standard loop
        ids_to_remove = []
        for face_id, data in self.face_tracker.items():
            if face_id not in matched_ids:
                data['frames'] -= 1
                if data['frames'] <= -10:
                    ids_to_remove.append(face_id)
        
        for face_id in ids_to_remove: 
            del self.face_tracker[face_id]
            
        return validated_faces

    def find_candidate(self, query_encoding):
        """Point 4 & 8: Mathematically sound matching without arbitrary ORB counts."""
        if not self.database or query_encoding is None:
            return None, 0
            
        best_match_name = None
        best_distance = 1.0 
        
        # Check against ALL aggregated encodings per identity
        for name, data in self.database.items():
            known_encodings = data.get("encodings", [])
            if not known_encodings:
                continue
            
            # Returns an array of Euclidean distances
            distances = face_recognition.face_distance(known_encodings, query_encoding)
            min_dist = np.min(distances)
            
            if min_dist < best_distance:
                best_distance = min_dist
                best_match_name = name
                
        # Point 4: Confidence formula relative to threshold
        if best_distance <= self.ENCODING_THRESHOLD:
            # 0.0 distance = 100% conf. At the threshold (e.g. 0.55), conf approaches 0%.
            confidence = max(0, (1.0 - (best_distance / self.ENCODING_THRESHOLD)) * 100)
            return best_match_name, confidence
            
        return None, 0

    def reconstruct_face(self, frame, x, y, w, h):
        face_roi = frame[y:y+h, x:x+w]
        
        # Enforce quality checks
        is_quality, quality_msg, _ = self.assess_face_quality(face_roi)
        if not is_quality:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, quality_msg, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame
        
        self.frame_count += 1
        
        rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        rgb_roi = self.apply_clahe(rgb_roi)  # Apply CLAHE preprocessing
        roi_h, roi_w = rgb_roi.shape[:2]
        split_y = int(roi_h * 0.55)
        
        # FIX: Removed the blackout strategy! 
        # Pass the natural (masked or unmasked) face crop directly to the encoder.
        encodings = face_recognition.face_encodings(rgb_roi, known_face_locations=[(0, roi_w, roi_h, 0)])
        
        if not encodings:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, "No Features", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return frame
            
        query_encoding = encodings[0]
        name, confidence = self.find_candidate(query_encoding)
        
        if name:
            display_name = name.split('_')[0]
            cand_data = self.database[name]
            
            if os.path.exists(cand_data["lower_path"]):
                lower_part_img = cv2.imread(cand_data["lower_path"])
                target_width = w
                scale_factor = target_width / lower_part_img.shape[1]
                new_height = int(lower_part_img.shape[0] * scale_factor)
                paste_y = y + split_y
                
                if paste_y < frame.shape[0]:
                    available_height = min(new_height, frame.shape[0] - paste_y)
                    available_width = min(target_width, frame.shape[1] - x)
                    
                    if available_height > 0 and available_width > 0:
                        resized_lower = cv2.resize(lower_part_img, (target_width, new_height))
                        resized_lower = resized_lower[:available_height, :available_width]
                        
                        try:
                            center = (x + available_width // 2, paste_y + available_height // 2)
                            mask = np.ones_like(resized_lower) * 255
                            
                            frame_roi = frame[paste_y:paste_y+available_height, x:x+available_width]
                            if frame_roi.shape[:2] == resized_lower.shape[:2]:
                                result_face = cv2.seamlessClone(resized_lower, frame, mask, center, cv2.NORMAL_CLONE)
                                frame[paste_y:paste_y+available_height, x:x+available_width] = \
                                    result_face[paste_y:paste_y+available_height, x:x+available_width]
                        except Exception:
                            frame_roi = frame[paste_y:paste_y+available_height, x:x+available_width]
                            result_face = cv2.addWeighted(frame_roi, 0.3, resized_lower, 0.7, 0)
                            frame[paste_y:paste_y+available_height, x:x+available_width] = result_face
                        
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {display_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {confidence:.0f}%", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame

    def get_frame(self):
        if self.video is None or not self.video.isOpened():
            for idx in [0, 1, 2]:
                self.video = cv2.VideoCapture(idx)
                if self.video.isOpened():
                    self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    break
            if not self.video.isOpened(): return None
        
        success, frame = self.video.read()
        if not success: return None
        
        faces = self.detect_faces_dnn(frame) if self.use_dnn else self.face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.05, minNeighbors=3, minSize=self.MIN_ROI_SIZE)
        
        validated_faces = [f for f in faces if self.validate_face_region(frame, *f)[0]]
        tracked_faces = self.track_faces(validated_faces)
        
        for (fx, fy, fw, fh) in tracked_faces[:1]:
            fw, fh = max(fw, self.MIN_ROI_SIZE[0]), max(fh, self.MIN_ROI_SIZE[1])
            fx, fy = max(0, min(fx, frame.shape[1] - fw)), max(0, min(fy, frame.shape[0] - fh))
            frame = self.reconstruct_face(frame, fx, fy, fw, fh)
            break
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def process_static_image(self, image_data):
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return None
        
        faces = self.detect_faces_dnn(frame) if self.use_dnn else self.face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.05, minNeighbors=3, minSize=self.MIN_ROI_SIZE)
        validated_faces = [f for f in faces if self.validate_face_region(frame, *f)[0]]
        
        for (fx, fy, fw, fh) in validated_faces:
            fw, fh = max(fw, self.MIN_ROI_SIZE[0]), max(fh, self.MIN_ROI_SIZE[1])
            fx, fy = max(0, min(fx, frame.shape[1] - fw)), max(0, min(fy, frame.shape[0] - fh))
            frame = self.reconstruct_face(frame, fx, fy, fw, fh)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()