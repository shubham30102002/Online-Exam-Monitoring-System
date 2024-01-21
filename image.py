import cv2

# Load the reference image
ref_img = cv2.imread('reference_image.jpg', cv2.IMREAD_GRAYSCALE)

# Create a feature detector
orb = cv2.ORB_create()

# Extract keypoints and descriptors from the reference image
ref_kp, ref_des = orb.detectAndCompute(ref_img, None)

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract keypoints and descriptors from the current frame
    cur_kp, cur_des = orb.detectAndCompute(gray, None)

    # Create a matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the descriptors of the reference image and the current frame
    matches = bf.match(ref_des, cur_des)

    # Sort the matches by their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the matches
    img_matches = cv2.drawMatches(ref_img, ref_kp, gray, cur_kp, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the image with matches
    cv2.imshow('Image Matches', img_matches)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()