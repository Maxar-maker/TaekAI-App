import eventlet
# IMPORTANT: patch before any other imports to avoid context errors
eventlet.monkey_patch()

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import os
import numpy as np
import base64
from pose_detector import PoseDetector
import logging
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/poses')

# Create static/poses directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    

# Initialize pose detector with higher threshold
pose_detector = PoseDetector(similarity_threshold=90.0)

def load_tutorial_poses():
    poses = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for img_file in sorted(os.listdir(app.config['UPLOAD_FOLDER'])):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file)
                keypoints = pose_detector.get_pose_landmarks(img_path)
                # Convert keypoints to regular numpy array if they exist
                if keypoints is not None and len(keypoints) > 0:
                    keypoints = keypoints.copy()  # Make a copy to ensure we have our own data
                    poses.append({
                        'landmarks': keypoints,
                        'filename': img_file
                    })
                    logger.info(f"Loaded pose image: {img_file}")
    return poses

# Initialize variables
tutorial_poses = []
current_pose_index = 0

# Load poses when the application starts
with app.app_context():
    tutorial_poses = load_tutorial_poses()

@app.route('/update_pose_index/<int:index>')
def update_pose_index(index):
    global current_pose_index
    if 0 <= index < len(tutorial_poses):
        current_pose_index = index
        # Reset state for new pose
        try:
            pose_detector.highest_similarity = 0.0
            pose_detector.best_frame = None
        except Exception:
            pass
        try:
            # Reset sustained-match counter if present
            if hasattr(handle_frame, 'match_counter'):
                handle_frame.match_counter = 0
        except Exception:
            pass
        return jsonify({'success': True, 'current_index': index})
    return jsonify({'success': False, 'error': 'Invalid index'})

@app.route('/check_pose_match')
def check_pose_match():
    global current_pose_index
    if current_pose_index < len(tutorial_poses):
        if pose_detector.current_keypoints is not None:
            similarity = pose_detector.calculate_pose_similarity(
                pose_detector.current_keypoints,
                tutorial_poses[current_pose_index]['landmarks']
            )
            # Convert numpy float32 to Python float
            similarity = float(similarity)
            if similarity >= pose_detector.similarity_threshold:
                return jsonify({'matched': True, 'similarity': similarity})
    return jsonify({'matched': False, 'similarity': 0.0})

def process_image(image_data):
    try:
        # Remove the data URL prefix (e.g., "data:image/jpeg;base64,")
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode image")
            return None
            
        # Process frame using pose detector
        if current_pose_index < len(tutorial_poses):
            frame, similarity, has_pose, best_frame = pose_detector.process_frame(
                frame,
                tutorial_poses[current_pose_index]['landmarks']
            )
        else:
            frame, _, has_pose, _ = pose_detector.process_frame(frame)
            
        # Encode the processed frame back to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/jpeg;base64,{encoded_image}"
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    # Send a test event to verify WebSocket communication
    emit('connection_test', {'message': 'WebSocket connection successful!'})
    
    # Send current pose information if available
    if current_pose_index < len(tutorial_poses):
        emit('current_pose_info', {
            'pose_index': current_pose_index,
            'pose_name': tutorial_poses[current_pose_index]['filename'],
            'total_poses': len(tutorial_poses)
        })

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('frame')
def handle_frame(data):
    """Handle incoming video frames from client"""
    try:
        # Skip every other frame for lighter processing (process only 2.5 FPS from 5 FPS)
        if not hasattr(handle_frame, 'frame_counter'):
            handle_frame.frame_counter = 0
        
        handle_frame.frame_counter += 1
        if handle_frame.frame_counter % 3 != 0:  # Process every 2nd frame
            return
        
        print(f"Processing frame {handle_frame.frame_counter}")
        
        # Check if data is the expected format
        if not isinstance(data, dict) or 'image' not in data:
            print(f"Error: Invalid data format received: {type(data)}")
            return
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("Error: Could not decode frame")
            return
        
        # Get pose detection results with overlayed similarity when target is known
        global pose_detector, tutorial_poses, current_pose_index
        reference_keypoints = None
        if 0 <= current_pose_index < len(tutorial_poses):
            reference_keypoints = tutorial_poses[current_pose_index].get('landmarks')

        processed_frame, similarity, has_person, best_frame = pose_detector.process_frame(
            frame, reference_keypoints
        )
        
        # Convert processed frame back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])  # Reduced quality further
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Emit processed frame back to client
        emit('processed_frame', {'image': f'data:image/jpeg;base64,{processed_base64}'})
        
        # Check for pose match using current detector state
        if pose_detector.current_keypoints is not None and reference_keypoints is not None:
            try:
                # Send real-time similarity updates for ALL similarity scores
                emit('pose_similarity', {
                    'similarity': float(similarity),
                    'pose_id': current_pose_index + 1,
                    'pose_name': tutorial_poses[current_pose_index]['filename']
                })

                # Confirm pose match when similarity meets configured threshold
                threshold = float(pose_detector.similarity_threshold)
                if float(similarity) >= threshold:
                    print(f"Pose match detected! Pose: {tutorial_poses[current_pose_index]['filename']}, Similarity: {similarity:.2f}")

                    # Immediately confirm match on first threshold hit for responsiveness
                    print(f"POSE MATCH CONFIRMED! Emitting event for {tutorial_poses[current_pose_index]['filename']}")
                    emit('pose_match_confirmed', {
                        'matched': True,
                        'pose_id': current_pose_index + 1,
                        'pose_name': tutorial_poses[current_pose_index]['filename'],
                        'similarity': float(similarity),
                        'message': f'Excellent! You matched {tutorial_poses[current_pose_index]["filename"]} with {similarity:.1f}% accuracy!'
                    })
                else:
                    # Reset match counter if similarity drops
                    if hasattr(handle_frame, 'match_counter'):
                        handle_frame.match_counter = 0
            except Exception as e:
                print(f"Error checking pose {current_pose_index}: {str(e)}")
        else:
            # Send status when no person detected
            emit('pose_similarity', {
                'similarity': 0,
                'message': 'No person detected'
            })
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': 'Error processing frame'})

def generate_frames():
    """Legacy function - kept for compatibility but not used in WebSocket version"""
    pass

@app.route('/video_feed')
def video_feed():
    """Legacy route - kept for compatibility but not used in WebSocket version"""
    return Response("WebSocket mode active", mimetype='text/plain')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Already monkey-patched at import time above
    env_port = os.environ.get('PORT')

    def pick_open_port(base_port: int, attempts: int = 20) -> int:
        """Find an open port starting at base_port, or return ephemeral."""
        for p in range(base_port, base_port + attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(('0.0.0.0', p))
                return p
            except OSError:
                continue
        # As a last resort, let OS assign an ephemeral port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', 0))
            return s.getsockname()[1]

    if env_port:
        # Respect explicit PORT (e.g., on PaaS). If busy, raise with guidance.
        port = int(env_port)
        try:
            socketio.run(app, host="0.0.0.0", port=port, debug=False)
        except OSError as e:
            logger.error(f"Port {port} is busy. Stop the other process or choose a different PORT.")
            raise
    else:
        # Local dev convenience: auto-pick a free port starting at 10000
        port = pick_open_port(10000)
        if port != 10000:
            logger.info(f"Port 10000 busy; using available port {port} instead.")
        socketio.run(app, host="0.0.0.0", port=port, debug=False)
