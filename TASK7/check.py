try:
    import face_recognition
    print(f"face_recognition version: {face_recognition.__version__}")  # Check if it has a version attribute
    print("face_recognition is installed correctly.")
except ImportError:
    print("face_recognition is not installed.")
except Exception as e:
    print(f"An error occurred: {e}")