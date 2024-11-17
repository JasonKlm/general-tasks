import asyncio
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed
from mavsdk.action import OrbitYawBehavior
import cv2
import cv2.aruco as aruco
import gi
import numpy as np

gi.require_version('Gst','1.0')
from gi.repository import Gst

markerLength = 10.0  # not entirely sure about the length of the marker
calib_path  = "./opencv/"
camera_matrix   = np.loadtxt(calib_path+'cameraMatrix_webcam.txt', delimiter=',')
camera_distortion   = np.loadtxt(calib_path+'cameraDistortion_webcam.txt', delimiter=',')
object_points = np.array([
    [-markerLength / 2, markerLength / 2, 0],
    [markerLength / 2, markerLength / 2, 0],
    [markerLength / 2, -markerLength / 2, 0],
    [-markerLength / 2, -markerLength / 2, 0]
], dtype=np.float32)

class Video():

    def __init__(self, port=5600):

        Gst.init(None)

        self.port = port
        self._frame = None

        self.video_source = 'udpsrc port = {}'.format(self.port)
        self.video_codec = '! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264'

        self.video_decode = \
            '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'

        self.video_sink_conf = \
            '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None
        
        self.run()

    def start_gst(self, config=None):

        if not config:
            config = \
                [
                    'videotestsrc ! decodebin',
                    '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                    '! appsink'
                ]

        command = ' '.join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name('appsink0')
        
    @staticmethod
    def gst_to_opencv(sample):


        buf = sample.get_buffer()
        caps = sample.get_caps()
        array = np.ndarray(
            (
                caps.get_structure(0).get_value('height'),
                caps.get_structure(0).get_value('width'),
                3
            ),
            buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)

        return array

    def frame(self):
        return self._frame

    def frame_available(self):
        return self._frame is not None

    def run(self):
        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf
            ])

        self.video_sink.connect('new-sample', self.callback)
        
    def callback(self, sink):
        
        sample = sink.emit('pull-sample')
        new_frame = self.gst_to_opencv(sample)
        self._frame = new_frame
        return Gst.FlowReturn.OK

def findMarkers(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    arucoParams = aruco.DetectorParameters()
    arucoDetector = aruco.ArucoDetector(arucoDict,arucoParams)
    
    bbox, ids, rejected = arucoDetector.detectMarkers(imgGray)
    
    if ids is not None:
        estimatePose(img,bbox,ids)
    
    return img, [bbox, ids]


def estimatePose(img, markerCorners, markerIds):
    # Incase there are multiple markers, loop through them to get poses
    for i in range(len(markerIds)):
        ret, rvec, tvec = cv2.solvePnP(object_points, markerCorners[i], camera_matrix, camera_distortion)
        
    cv2.drawFrameAxes(img, camera_matrix, camera_distortion, rvec, tvec, markerLength)
    print("X:", tvec[0], "Y:", tvec[1], "Z:", tvec[2])
     
async def main():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone!")
            break

    print("--Arming")
    await drone.action.arm()

    print("--Taking off")
    await drone.action.takeoff()

    await asyncio.sleep(11)
    
    # not using offboard mode, drifting issues
    
    #initial_velocity = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    #await drone.offboard.set_velocity_body(initial_velocity)

    # print("-- Setting offboard mode")
    # await drone.offboard.start()

    async for position in drone.telemetry.position():
        latitude = position.latitude_deg
        longitude = position.longitude_deg
        altitude = position.absolute_altitude_m
        break
    
    
    print("Orbit") 
    await drone.action.do_orbit(5, 1, OrbitYawBehavior.HOLD_FRONT_TO_CIRCLE_CENTER, latitude, longitude, altitude)

    await asyncio.sleep(6)
    
    video = Video()

    while True:
        
        if video.frame_available():
            frame = video.frame()
            copy_frame = frame.copy()
            detected_frame, arucoFound = findMarkers(copy_frame)
                
            cv2.imshow("Drone Camera Stream with ArUco Detection", detected_frame)
            if len(arucoFound[0]) != 0:
                print("Tag detected.")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            await asyncio.sleep(0.1)
        


if __name__ == "__main__":
    asyncio.run(main())

