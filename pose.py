import asyncio
import numpy as np
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed
import cv2
import cv2.aruco as aruco

markerLength = 10.0
calib_path = "./opencv/"
camera_matrix = np.loadtxt(calib_path + "cameraMatrix_webcam.txt", delimiter=",")
camera_distortion = np.loadtxt(
    calib_path + "cameraDistortion_webcam.txt", delimiter=","
)
# Define object points for the ArUco marker (in 3D space)
object_points = np.array([
    [-markerLength / 2, markerLength / 2, 0],
    [markerLength / 2, markerLength / 2, 0],
    [markerLength / 2, -markerLength / 2, 0],
    [-markerLength / 2, -markerLength / 2, 0]
], dtype=np.float32)

async def setup_drone():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone!")
            break

    print("-- Arming")
    await drone.action.arm()
    print("-- Taking off")
    await drone.action.takeoff()
    await asyncio.sleep(10)  # Wait for the drone to stabilize in the air
    return drone

def rotation_matrix(yaw, pitch, roll):
    # Convert angles from degrees to radians
    yaw, pitch, roll = map(np.radians, [yaw, pitch, roll])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

    return Rz @ Ry @ Rx  # Combined rotation matrix

async def get_drone_position_and_orientation(drone):
    async for position in drone.telemetry.position():
        async for euler_angle in drone.telemetry.attitude_euler():
            return {
                "position": {
                    "latitude": position.latitude_deg,
                    "longitude": position.longitude_deg,
                    "altitude": position.absolute_altitude_m,
                },
                "orientation": {
                    "yaw": euler_angle.yaw_deg,
                    "pitch": euler_angle.pitch_deg,
                    "roll": euler_angle.roll_deg,
                }
            }

def estimate_marker_pose(frame):
    # Convert frame to grayscale for marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    arucoDetector = aruco.ArucoDetector(aruco_dict, parameters)
    
    # Detect markers
    corners, ids, _ = arucoDetector.detectMarkers(gray)
    if ids is not None:
        # Estimate pose of each marker
        rvec, tvec, _ = cv2.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, camera_distortion)
        return tvec[0][0], rvec[0][0]  # Returning the first detected marker's pose
    return None, None

def transform_marker_to_global(tvec, drone_position, drone_orientation):
    # Convert the drone's orientation to a rotation matrix
    R_global = rotation_matrix(
        drone_orientation["yaw"], drone_orientation["pitch"], drone_orientation["roll"]
    )
    
    # Apply the transformation to get the global position of the marker
    tvec_global = R_global @ tvec.T + np.array([
        [drone_position["latitude"]],
        [drone_position["longitude"]],
        [drone_position["altitude"]]
    ])
    
    return tvec_global.flatten()

async def main():
    # Step 1: Setup drone
    drone = await setup_drone()

    # Step 2: Capture video from the camera
    cap = cv2.VideoCapture(0)  # Adjust based on your camera setup

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Step 3: Detect marker and estimate its pose relative to the camera
        tvec, rvec = estimate_marker_pose(frame)
        if tvec is not None:
            # Step 4: Get drone's global position and orientation
            telemetry = await get_drone_position_and_orientation(drone)
            drone_position = telemetry["position"]
            drone_orientation = telemetry["orientation"]

            # Step 5: Transform marker position to global coordinates
            marker_global_position = transform_marker_to_global(tvec, drone_position, drone_orientation)
            print(f"Global Position of marker: Latitude={marker_global_position[0]}, Longitude={marker_global_position[1]}, Altitude={marker_global_position[2]}")

            # Step 6: Navigate to marker position
            print("Moving to the detected marker's position...")
            await drone.action.goto_location(
                marker_global_position[0], marker_global_position[1], marker_global_position[2], drone_orientation["yaw"]
            )
            await asyncio.sleep(1)

        # Display the frame with detected markers
        cv2.imshow("Drone Camera Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
