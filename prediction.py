import cv2
import tensorflow as tf
import numpy as np
import time

# Load and recompile the model
model = tf.keras.models.load_model('M:/PROJECTS/AI_PROJECT/DATASET/DATASET/final_ai_model.h5')

class_names = ['BALL_DEFECT', 'CAGE_DEFECT', 'HEALTHY', 'INNER_RACE_DEFECT', 'LACK_OF_LUBRICATION', 'OUTER_RACE_DEFECT']

def realtime_video_predict(video_path):
    # Open the test video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # For FPS calculation
    prev_time = 0
    curr_time = 0
    
    print(f"Starting real-time prediction from video file: {video_path}")
    print("Press 'q' to quit, or 'p' to pause/resume")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # Preprocess frame
        img = cv2.resize(frame, (224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        pred = model.predict(img_array, verbose=0)
        confidence = np.max(pred)
        predicted_class = class_names[np.argmax(pred)]
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Determine text color based on prediction
        if predicted_class == 'HEALTHY':
            text_color = (0, 255, 0)  # Green for healthy
        else:
            text_color = (0, 0, 255)  # Red for defects
        
        # Get frame dimensions
        frame_height, frame_width = display_frame.shape[:2]
        
        # Display frame number at top with transparent background
        frame_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Center the frame counter text at top
        frame_text = f"Frame: {frame_position}/{total_frames}"
        text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x_position = (frame_width - text_size[0]) // 2
        
        # Create transparent background for top text
        overlay_top = display_frame.copy()
        cv2.rectangle(overlay_top, 
                     (x_position - 5, 5), 
                     (x_position + text_size[0] + 5, 25), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay_top, 0.5, display_frame, 0.5, 0, display_frame)
        
        cv2.putText(display_frame, frame_text, 
                   (x_position, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add prediction info at bottom with transparent background
        # Use larger text size for better visibility but still not dominating
        font_size = 0.6
        line_spacing = 25
        
        # Start position for bottom text (slightly above bottom)
        y_base = frame_height - 30
        
        # Create transparent background for bottom left text (predictions)
        bottom_overlay = display_frame.copy()
        
        # Calculate width needed for prediction text
        pred_text = f"Pred: {predicted_class}"
        conf_text = f"Conf: {confidence:.2%}"
        pred_width = max(
            cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)[0][0],
            cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)[0][0]
        ) + 10
        
        # Draw background for prediction and confidence
        cv2.rectangle(bottom_overlay, 
                     (5, y_base - 20), 
                     (pred_width, y_base + line_spacing + 5),
                     (0, 0, 0), -1)
        
        # FPS text (right side)
        fps_text = f"FPS: {fps:.1f}"
        fps_text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)[0]
        
        # Background for FPS counter
        cv2.rectangle(bottom_overlay, 
                    (frame_width - fps_text_size[0] - 15, y_base + line_spacing - 20),
                    (frame_width - 5, y_base + line_spacing + 5),
                    (0, 0, 0), -1)
        
        # Apply the overlay with transparency
        cv2.addWeighted(bottom_overlay, 0.5, display_frame, 0.5, 0, display_frame)
        
        # Prediction text
        cv2.putText(display_frame, pred_text, 
                  (10, y_base), 
                  cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, 1)
        
        # Confidence text
        cv2.putText(display_frame, conf_text, 
                  (10, y_base + line_spacing), 
                  cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, 1)
        
        # FPS text
        cv2.putText(display_frame, fps_text, 
                  (frame_width - fps_text_size[0] - 10, y_base + line_spacing), 
                  cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow('Bearing Defect Prediction', display_frame)
        
        # Get key press
        key = cv2.waitKey(25) & 0xFF
        
        # Exit on 'q' key press
        if key == ord('q'):
            break
        # Pause/resume on 'p' key press
        elif key == ord('p'):
            print("Video paused. Press any key to continue...")
            cv2.waitKey(-1)  # Wait indefinitely until a key is pressed
        # Save snapshot on 's' key press
        elif key == ord('s'):
            snapshot_file = f"bearing_defect_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(snapshot_file, display_frame)
            print(f"Snapshot saved as {snapshot_file}")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use the path to your test video
    video_path = r'M:/PROJECTS/AI_PROJECT/TEST_VIDEO/lack_of_lubrication_video.mp4'
    realtime_video_predict(video_path)
