import cv2
import numpy as np
import csv
from datetime import timedelta

def track_balls(video_path, output_video_path, output_csv_path):
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    color_ranges = {
        'red': ([0, 100, 100], [10, 255, 255], (0, 0, 255)),
        'yellow': ([20, 100, 100], [30, 255, 255], (0, 255, 255)),
        'green': ([40, 100, 100], [80, 255, 255], (0, 255, 0)),
        'white': ([0, 0, 200], [180, 30, 255], (255, 255, 255))
    }
    
    ball_trackers = {color: {} for color in color_ranges}
    events = []
    
    quadrants = {
        1: (0, 0, width//2, height//2),
        2: (width//2, 0, width, height//2),
        3: (0, height//2, width//2, height),
        4: (width//2, height//2, width, height)
    }
    
    # Background subtraction
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        timestamp = frame_number / fps
        
        # Apply background subtraction
        fg_mask = backSub.apply(frame)
        
        # Apply some morphological operations to remove noise
        kernel = np.ones((5,5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for color, (lower, upper, bgr_color) in color_ranges.items():
            if color == 'red':
                lower_red = np.array([0, 100, 100])
                upper_red = np.array([10, 255, 255])
                mask1 = cv2.inRange(hsv, lower_red, upper_red)
                lower_red = np.array([160, 100, 100])
                upper_red = np.array([180, 255, 255])
                mask2 = cv2.inRange(hsv, lower_red, upper_red)
                mask = mask1 + mask2
            elif color == 'white':
                # For white, we'll use the foreground mask directly
                mask = fg_mask
            else:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Combine color mask with foreground mask
            mask = cv2.bitwise_and(mask, fg_mask)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Adjust this threshold as needed
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        cv2.circle(frame, (cx, cy), 10, bgr_color, 2)
                        
                        current_quadrant = None
                        for q, (x, y, w, h) in quadrants.items():
                            if x <= cx < w and y <= cy < h:
                                current_quadrant = q
                                break
                        
                        if current_quadrant:
                            if color not in ball_trackers:
                                ball_trackers[color] = {}
                            
                            if current_quadrant not in ball_trackers[color]:
                                event = [timestamp, current_quadrant, color, "Entry"]
                                events.append(event)
                                ball_trackers[color][current_quadrant] = True
                                cv2.putText(frame, f"Entry Q{current_quadrant} {timedelta(seconds=timestamp)}", 
                                            (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
                            elif ball_trackers[color][current_quadrant] is False:
                                event = [timestamp, current_quadrant, color, "Entry"]
                                events.append(event)
                                ball_trackers[color][current_quadrant] = True
                                cv2.putText(frame, f"Entry Q{current_quadrant} {timedelta(seconds=timestamp)}", 
                                            (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
                            
                            for q in quadrants:
                                if q != current_quadrant and q in ball_trackers[color] and ball_trackers[color][q]:
                                    event = [timestamp, q, color, "Exit"]
                                    events.append(event)
                                    ball_trackers[color][q] = False
                                    cv2.putText(frame, f"Exit Q{q} {timedelta(seconds=timestamp)}", 
                                                (cx, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time", "Quadrant Number", "Ball Colour", "Type (Entry or Exit)"])
        for event in events:
            writer.writerow([str(timedelta(seconds=event[0])), event[1], event[2], event[3]])

    print(f"Event data has been saved to {output_csv_path}")
    print(f"Processed video has been saved to {output_video_path}")

# Example usage
video_path = "/content/drive/MyDrive/AI Assignment video.mp4"
output_video_path = "processed_video.mp4"
output_csv_path = "ball_tracking_events.csv"
track_balls(video_path, output_video_path, output_csv_path)
