from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera
import train_model
import os
import base64
import time
import shutil
import threading

app = Flask(__name__)
video_camera = None
training_lock = threading.Lock()

def get_camera():
    global video_camera
    if video_camera is None:
        video_camera = VideoCamera()
    return video_camera

@app.route('/stop_camera')
def stop_camera():
    global video_camera
    if video_camera:
        video_camera.release()
        video_camera = None
    return jsonify({"status": "success", "message": "Camera released"})

@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', active_page='home', error="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({"status": "error", "message": "Internal Server Error"}), 500

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html', active_page='home')

@app.route('/train_page')
def train_page():
    return render_template('train.html', active_page='train')

@app.route('/dataset_page')
def dataset_page():
    return render_template('dataset.html', active_page='dataset')

@app.route('/photo_page')
def photo_page():
    return render_template('photo.html', active_page='photo')

@app.route('/video_feed')
def video_feed():
    return Response(gen(get_camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train', methods=['POST'])
def train():
    if training_lock.locked():
        return jsonify({"status": "error", "message": "Training already in progress"}), 429
        
    with training_lock:
        try:
            train_model.train_dataset()
            get_camera().reload_database()
            return jsonify({"status": "success", "message": "Model trained and reloaded successfully!"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/dataset')
def get_dataset():
    import os
    dataset_dir = "dataset_unmasked"
    if not os.path.exists(dataset_dir):
        return jsonify({})
    
    files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    grouped_files = {}
    for file in files:
        name = file.split('_')[0]
        if name not in grouped_files:
            grouped_files[name] = []
        grouped_files[name].append(file)
        
    return jsonify(grouped_files)

@app.route('/dataset/<filename>')
def serve_dataset_image(filename):
    from flask import send_from_directory
    return send_from_directory('dataset_unmasked', filename)

@app.route('/capture', methods=['POST'])
def capture():
    try:
        data = request.json
        image_data = data['image']
        name = data['name']
        
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        os.makedirs("dataset_unmasked", exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join("dataset_unmasked", filename)
        
        with open(filepath, "wb") as f:
            f.write(binary_data)
            
        return jsonify({"status": "success", "message": "Image captured successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/reset_data', methods=['POST'])
def reset_data():
    try:
        dirs = ["dataset_unmasked", "db_upper", "db_lower"]
        for d in dirs:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
            
        if os.path.exists("features_db.pkl"):
            os.remove("features_db.pkl")
            
        get_camera().reload_database()
        return jsonify({"status": "success", "message": "All data reset successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/detect_photo', methods=['POST'])
def detect_photo():
    try:
        data = request.json
        image_data = data['image']
        
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        
        processed_bytes = get_camera().process_static_image(binary_data)
        
        if processed_bytes is None:
             return jsonify({"status": "error", "message": "Could not process image"}), 400

        processed_base64 = base64.b64encode(processed_bytes).decode('utf-8')
        return jsonify({"status": "success", "image": f"data:image/jpeg;base64,{processed_base64}"})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)