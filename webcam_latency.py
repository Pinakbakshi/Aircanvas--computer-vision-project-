import cv2
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set resolution (optional, based on your webcam capabilities)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Measuring latency... Press 'q' to stop.")

# Initialize variables
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame capture failed.")
        break

    # Increment frame count
    frame_count += 1

    # Display the frame (optional, can be removed for latency-only testing)
    cv2.imshow("Webcam Latency Test", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate total time and frames per second
end_time = time.time()
elapsed_time = end_time - start_time
fps = frame_count / elapsed_time

# Display results
print(f"Total frames captured: {frame_count}")
print(f"Total time elapsed: {elapsed_time:.2f} seconds")
print(f"Approximate FPS (frames per second): {fps:.2f}")

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
