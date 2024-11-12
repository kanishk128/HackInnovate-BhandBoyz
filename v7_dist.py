import cv2
import streamlit as st
import time
import torch
from gtts import gTTS
import os
from datetime import datetime
import numpy as np
from moviepy.editor import VideoFileClip

# Streamlit app title and description
st.title("🔍 VisionSaathi by Team Bhandboyz")
st.sidebar.header("Settings")
st.sidebar.write("Configure the options for the app below.")

# Toggle for audio alerts
audio_alert = st.sidebar.checkbox("Enable Audio Alerts", value=True)

# Placeholder for video feed and detection history
video_placeholder = st.empty()
detection_history_container = st.container()  # Container for showing detection history

# Initialize detection history if not already done
if "detection_texts" not in st.session_state:
    st.session_state.detection_texts = []

# Function to trim the video based on user-defined start and end times
def trim_video(input_path, start=0, end=20):
    output_path = "shortened_video.mp4"
    clip = VideoFileClip(input_path)
    
    # Check if video length is less than 20 seconds
    if clip.duration < 20:
        # Save the original video as the output
        clip.write_videofile(output_path, codec="libx264")
    else:
        # Trim video if duration is greater than or equal to 20 seconds
        trimmed_clip = clip.subclip(start, min(end, clip.duration))  # Ensure end does not exceed video duration
        trimmed_clip.write_videofile(output_path, codec="libx264")
        trimmed_clip.close()
        
    clip.close()
    return output_path

# Create a 'recordings' folder if it doesn't exist
if not os.path.exists('recordings'):
    os.makedirs('recordings')

# Upload the video file
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
if uploaded_video:
    # Save the uploaded file
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_video.getbuffer())
    
    video = VideoFileClip("uploaded_video.mp4")
    video_duration = video.duration  # in seconds
    video.close()

    # User input for start and end time
    start_time = st.sidebar.number_input("Start Time (in seconds)", min_value=0, value=0)
    if video_duration < 20:
        end_time = st.sidebar.number_input("End Time (in seconds)", min_value=0, value=1)
    else:
        end_time = st.sidebar.number_input("End Time (in seconds)", min_value=0, value=20)
    
    # Check if start and end time are valid; otherwise, use default 20 seconds duration
    if start_time >= end_time:
        st.warning("End time should be greater than start time. Using default duration.")
        end_time = start_time + 20  # Default 20 seconds

    # Trim the video
    trimmed_video_path = trim_video("uploaded_video.mp4", start=start_time, end=end_time)
    
    # Load the trimmed video for further processing
    cap = cv2.VideoCapture(trimmed_video_path)

    # Load YOLOv7 model (ensure 'yolov7.pt' path is correct)
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model='yolov7.pt', source='github')

    # Function to calculate distance based on bounding box width
    def calculate_distance(width_in_frame, actual_width=0.5, focal_length=615):
        """Estimate distance to object in meters."""
        if width_in_frame > 0:
            return (actual_width * focal_length) / width_in_frame
        return None

    # Check if video source is accessible
    if not cap.isOpened():
        st.error("Unable to open the trimmed video.")
    else:
        last_capture_time = time.time()

        # Stream frames from the video feed
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to retrieve frame.")
                break

            # Convert frame to RGB for display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display_frame, channels="RGB")

            # Check if 10 seconds have passed for the next capture
            if time.time() - last_capture_time >= 2:
                last_capture_time = time.time()

                # Perform object detection on the current frame
                results = model(frame)
                detections = results.pred[0].cpu().numpy()  # Get predictions
                names = results.names  # Access class names from model

                detection_entries = []  # To store detection texts for audio and display

                # Define the frame's width and boundaries for segmentation
                frame_width = frame.shape[1]
                left_boundary = frame_width // 3
                right_boundary = left_boundary * 2

                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    if conf > 0.5:  # Only consider high-confidence detections
                        obj_name = names[int(cls)]
                        width_in_frame = x2 - x1

                        # Calculate distance to the detected object
                        distance = calculate_distance(width_in_frame)
                        detected_text = f"{obj_name} at {distance:.2f}m" if distance else f"{obj_name} (distance unknown)"

                        # Determine object position (left, middle, or right)
                        center_x = (x1 + x2) / 2
                        if center_x < left_boundary:
                            position = "on your left"
                        elif center_x < right_boundary:
                            position = "in front of you"
                        else:
                            position = "on your right"
                        
                        # Add position to detection text
                        detected_text += f" {position}"
                        detection_entries.append(detected_text)

                        # Draw bounding box and distance on frame
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"{obj_name}: {distance:.2f}m {position}" if distance else f"{obj_name}: Calculating... {position}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Display the updated frame in Streamlit
                video_placeholder.image(frame, channels="BGR")

                if detection_entries:
                    # Add timestamp to detections
                    detection_time = datetime.now().strftime("%H:%M:%S")
                    detection_text = f"{detection_time} - Detected: {', '.join(detection_entries)}"

                    # Update detection history in session state and limit to last 5 entries
                    st.session_state.detection_texts.append(detection_text)
                    st.session_state.detection_texts = st.session_state.detection_texts[-5:]

                    detection_text = f"Detected: {', '.join(detection_entries)}"

                    # Display detection history
                    with detection_history_container:
                        detection_history_container.empty()
                        st.markdown("### Detection History")
                        for text in reversed(st.session_state.detection_texts):
                            st.write(text)

                    # Generate unique audio message if enabled
                    if audio_alert:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        audio_filename = f"recordings/detection_{timestamp}.mp3"
                        tts = gTTS(detection_text)
                        tts.save(audio_filename)

                        # Play the generated audio file
                        os.system(f"start {audio_filename}")  # Adjust as needed for Mac/Linux

                        # Delay the next processing by twice the length of the audio file
                        audio_duration = len(detection_text) / 10  # Approximate duration of audio (one character ~0.1 sec)
                        time.sleep(audio_duration * 2)  # Wait for twice the audio length before processing the next frame

    # Release resources
    cap.release()
else:
    st.info("Please upload a video file to begin.")
