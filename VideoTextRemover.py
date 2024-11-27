import cv2
import os
import numpy as np
import re 
import time
import shutil

def sortedproper(l):    
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def detect_subtitles(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Try multiple thresholding methods
    methods = [
        cv2.THRESH_BINARY,
        cv2.THRESH_BINARY_INV,
        cv2.THRESH_OTSU
    ]
    
    for method in methods:
        # Try different threshold values
        for thresh in [230, 240, 250]:
            # Threshold the image
            _, binary = cv2.threshold(gray, thresh, 255, method)
            
            # Morphological operations to improve detection
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.dilate(binary, kernel, iterations=3)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours based on size and position
            subtitle_contours = [
                cnt for cnt in contours 
                if (cv2.contourArea(cnt) > 100 and  # Minimum area
                    cv2.contourArea(cnt) < frame.shape[0] * frame.shape[1] * 0.3)  # Maximum area
            ]
            
            if subtitle_contours:
                # Create mask
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, subtitle_contours, -1, 255, -1)
                return mask
    
    # If no subtitles detected
    return np.zeros(gray.shape, dtype=np.uint8)

def remove_subtitles(input_video_path, output_video_path=None, temp_dir=None):
    """
    Remove subtitles from a video
    
    Parameters:
    -----------
    input_video_path : str
        Path to the input video file
    output_video_path : str, optional
        Path to save the output video. If None, will be generated based on input path
    temp_dir : str, optional
        Directory to store temporary frames. If None, will use a temp directory in the same location as input video
    
    Returns:
    --------
    str
        Path to the output video file
    """
    # Start timing
    start_time = time.time()
    
    # Validate input video
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")
    
    # Determine output and temp directories
    input_dir = os.path.dirname(input_video_path)
    input_filename = os.path.splitext(os.path.basename(input_video_path))[0]
    
    # Set output video path
    if output_video_path is None:
        output_video_path = os.path.join(input_dir, f"{input_filename}_no_subtitles.webm")
    
    # Set temp directory
    if temp_dir is None:
        temp_dir = os.path.join(input_dir, f"{input_filename}_temp")
    
    # Create temp directory
    os.makedirs(temp_dir, exist_ok=True)
    
    # Video capture
    vid = cv2.VideoCapture(input_video_path)
    
    # Process frames
    frame_counter = 0
    processed_frames = []
    
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # Detect subtitles
        subtitle_mask = detect_subtitles(frame)
        
        # Inpaint to remove subtitles
        if np.sum(subtitle_mask) > 0:
            cleaned_frame = cv2.inpaint(frame, subtitle_mask, 3, cv2.INPAINT_TELEA)
        else:
            cleaned_frame = frame

        # Save frame
        filename = os.path.join(temp_dir, f"frame{frame_counter}.jpg")
        cv2.imwrite(filename, cleaned_frame)
        processed_frames.append(filename)
        frame_counter += 1

    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    output_video = cv2.VideoWriter(output_video_path, fourcc, 
        int(vid.get(cv2.CAP_PROP_FPS)), 
        (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), 
         int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    # Write processed frames
    sorted_frames = sortedproper([f for f in os.listdir(temp_dir) if f.endswith(".jpg")])
    for frame_filename in sorted_frames:
        frame_path = os.path.join(temp_dir, frame_filename)
        output_video.write(cv2.imread(frame_path))

    # Cleanup
    output_video.release()
    vid.release()
    
    # Remove temporary directory
    try:
        shutil.rmtree(temp_dir) # Remove the temporary directory and its contents
        print(f"Temporary directory '{temp_dir}' deleted successfully.")
    except OSError as e:
        print(f"Error deleting temporary directory: {e}") # Handle potential errors


    print(f"Subtitle removal completed in {time.time() - start_time:.2f} seconds")
    print(f"Output video saved to: {output_video_path}")
    
    return output_video_path

# Example usage
if __name__ == "__main__":
    input_video = 'test/input.mp4'
    remove_subtitles(input_video)