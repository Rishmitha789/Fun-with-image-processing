import cv2
import numpy as np
from scipy.fft import rfft, rfftfreq, irfft
import matplotlib.pyplot as plt

# Step 2: Read and collect green channel from video frames
video = '"C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\Pulse\\finger.mp4"'
cap = cv2.VideoCapture(video)
#
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

green_channel_data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    green_channel = frame[:, :, 1]  # Extract the green channel (0: Blue, 1: Green, 2: Red)
    green_channel_data.extend(green_channel.flatten())

cap.release()

# Check if the data is empty or too small for FFT
if not green_channel_data:
    print("No valid data found in the video.")
else:
    # Continue with signal processing
    # ... (Rest of the code)

# ... (Rest of the code)


    # Step 3: Mean normalize and visualize the signal
    mean_value = np.mean(green_channel_data)
    normalized_signal = np.array(green_channel_data) - mean_value

    plt.figure(figsize=(12, 4))
    plt.plot(normalized_signal)
    plt.title('Normalized Signal')
    plt.xlabel('Frame')
    plt.ylabel('Pixel Value')
    plt.show()

    # Step 4: Apply Fast Fourier Transform (FFT) to the signal
    signal_fft = rfft(normalized_signal)

    # Step 5: Filter out unwanted frequencies and visualize
    frequency = rfftfreq(len(normalized_signal), 1 / fps)
    low_freq_limit = 0.45  # Minimum human pulse frequency (27 BPM)
    high_freq_limit = 8.0  # Maximum human pulse frequency (480 BPM)

    # Filter out frequencies outside the pulse range
    signal_fft[(frequency < low_freq_limit) | (frequency > high_freq_limit)] = 0

    # Step 6: Calculate heart rate (BPM)
    max_amplitude_freq = frequency[np.argmax(np.abs(signal_fft))]
    heart_rate_bpm = max_amplitude_freq * 60

    print(f'Estimated Heart Rate: {heart_rate_bpm:.2f} BPM')
printf('')