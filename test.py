import cv2
import cv2.aruco as aruco

def findMarkers(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    arucoParams = aruco.DetectorParameters()
    arucoDetector = aruco.ArucoDetector(arucoDict, arucoParams)
    
    bbox, ids, rejected = arucoDetector.detectMarkers(imgGray)
    
    if ids is not None:
        aruco.drawDetectedMarkers(img, bbox, ids)
        print(f"Detected ArUco IDs: {ids.flatten()}")
    else:
        print("No ArUco markers detected.")
    
    return img

def main():
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        detected_frame = findMarkers(frame)
        cv2.imshow("Webcam ArUco Detection", detected_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()