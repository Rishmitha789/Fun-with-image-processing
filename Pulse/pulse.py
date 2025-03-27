import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft  # For Fourier Transform

video_path = "./After climbbing.mp4"  

# Open the video file
vid = cv2.VideoCapture(video_path)

if not vid.isOpened():
    print("Error: Cannot open video file.")
    exit()

fps = vid.get(cv2.CAP_PROP_FPS)  # Get video frame rate
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
video_duration = frame_count / fps  # Video duration in seconds

green_means = []  # Store mean green intensity per frame

while True:
    ret, frame = vid.read()
    if not ret:
        break  # Exit when video ends

    green_channel = frame[:, :, 1]  # Extract green channel
    green_means.append(green_channel.mean())  # Store mean intensity of frame

vid.release()  # Release video capture

# Convert list to NumPy array
green_means = np.array(green_means)

# Compute mean normalization
mean = np.mean(green_means)
normalized_signal = green_means - mean

# Plot the normalized signal over frames
plt.figure(figsize=(12, 6))
plt.plot(normalized_signal, color='b', label="Mean Normalized Green Signal")
plt.title("Mean Normalized Green Channel Signal")
plt.xlabel("Frame Index")
plt.ylabel("Normalized Pixel Value")
plt.grid(True)
plt.legend()
plt.show()

# **Apply Fourier Transform (FFT) on the 1D mean signal**
fourier_signals = rfft(normalized_signal)
freqs = rfftfreq(len(normalized_signal), d=1 / fps)

# Plot the FFT Magnitude Spectrum
plt.figure(figsize=(12, 6))
plt.plot(freqs, np.abs(fourier_signals), color='r', label="FFT Magnitude Spectrum")
plt.title("FFT of Normalized Green Channel Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.legend()
plt.show()

# **Apply frequency filter (0.45 Hz to 8 Hz)**
low_cutoff = 0.45  # 27 BPM
high_cutoff = 8.0  # 480 BPM

filtered_fourier = fourier_signals.copy()
filtered_fourier[(freqs < low_cutoff) | (freqs > high_cutoff)] = 0  # Remove unwanted frequencies

# Apply inverse FFT to reconstruct the filtered signal
filtered_signal = irfft(filtered_fourier)

# **Find the dominant frequency (peak amplitude)**
dominant_freq = freqs[np.argmax(np.abs(filtered_fourier))]  # Frequency with highest amplitude

# Calculate estimated pulse rate in BPM
estimated_bpm = dominant_freq * 60  # Convert Hz to BPM

# **Plot the filtered signal**
plt.figure(figsize=(12, 6))
plt.plot(filtered_signal, color='g', label="Filtered Signal (Pulse Range)")
plt.title("Filtered Green Channel Signal (0.45 Hz to 8 Hz)")
plt.xlabel("Frame Index")
plt.ylabel("Filtered Pixel Value")
plt.grid(True)
plt.legend()
plt.show()

# **Print results**
print(f"Estimated Pulse Rate: {estimated_bpm:.2f} BPM")
print(f"Dominant Frequency: {dominant_freq:.2f} Hz")
print(f"Video Duration: {video_duration:.2f} seconds")

cv2.destroyAllWindows()
