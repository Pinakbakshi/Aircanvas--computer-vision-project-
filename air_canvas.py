import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Drawing parameters
drawing_color = (255, 0, 0)  # Default color is red
drawing_radius = 5
canvas = None
last_position = None  # Stores the previous finger position for smooth lines

# Gesture threshold
gesture_threshold = 30  # Distance (pixels) between thumb and index to trigger drawing

# Color selection coordinates
color_palette = {
    "red": ((50, 50), (150, 150), (0, 0, 255)),  # Red button
    "green": ((200, 50), (300, 150), (0, 255, 0))  # Green button
}

# Quit button coordinates
quit_pane = {"quit": ((500, 50), (600, 100))}  # Top-right pane
show_quit_confirmation = False

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set maximum resolution and frame rate for smooth performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)  # Set the frame rate to 60 FPS (if supported)

# Initialize Mediapipe hands with optimized settings
with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(rgb_frame)

        # Initialize canvas if it's None
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Draw the color selection palette
        for color, (start, end, color_bgr) in color_palette.items():
            cv2.rectangle(frame, start, end, color_bgr, -1)
            cv2.putText(frame, color.capitalize(), (start[0] + 10, end[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw Quit pane
        cv2.rectangle(frame, quit_pane["quit"][0], quit_pane["quit"][1], (0, 0, 0), -1)
        cv2.putText(frame, "Quit", (quit_pane["quit"][0][0] + 10, quit_pane["quit"][0][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # If the quit confirmation is shown
        if show_quit_confirmation:
            cv2.rectangle(frame, (500, 110), (600, 160), (0, 0, 0), -1)
            cv2.putText(frame, "Yes", (510, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(frame, (500, 170), (600, 220), (0, 0, 0), -1)
            cv2.putText(frame, "No", (510, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates of the index finger and thumb
                h, w, _ = frame.shape
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                index_pos = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))

                # Check if thumb and index finger are close enough to trigger drawing
                distance = np.sqrt((thumb_pos[0] - index_pos[0]) ** 2 + (thumb_pos[1] - index_pos[1]) ** 2)
                if distance < gesture_threshold:
                    # Draw on the canvas
                    if last_position is not None:
                        cv2.line(canvas, last_position, index_pos, drawing_color, thickness=5)  # Draw continuous lines
                    last_position = index_pos
                else:
                    last_position = None  # Reset when gesture is not active

                # Check if finger is in any color selection box
                for color, (start, end, color_bgr) in color_palette.items():
                    if start[0] <= index_pos[0] <= end[0] and start[1] <= index_pos[1] <= end[1]:
                        drawing_color = color_bgr  # Change drawing color
                        cv2.putText(frame, f"Selected {color.capitalize()}!",
                                    (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 2)

                # Check if finger is in Quit pane
                if (quit_pane["quit"][0][0] <= index_pos[0] <= quit_pane["quit"][1][0] and
                        quit_pane["quit"][0][1] <= index_pos[1] <= quit_pane["quit"][1][1]):
                    show_quit_confirmation = True

                # If quit confirmation is active, check "Yes" or "No"
                if show_quit_confirmation:
                    if 500 <= index_pos[0] <= 600 and 110 <= index_pos[1] <= 160:  # "Yes"
                        cap.release()
                        cv2.destroyAllWindows()
                        exit(0)  # Quit the application
                    if 500 <= index_pos[0] <= 600 and 170 <= index_pos[1] <= 220:  # "No"
                        show_quit_confirmation = False

        # Combine canvas and original frame
        combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        # Display the combined image
        cv2.imshow("Air Canvas with Improved Frame Rate", combined)

        # Check for user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Clear the canvas
            canvas = np.zeros_like(frame)
        elif key == ord('s'):  # Save the drawing
            cv2.imwrite("drawing.png", canvas)
            print("Drawing saved as 'drawing.png'!")

cap.release()
cv2.destroyAllWindows()
