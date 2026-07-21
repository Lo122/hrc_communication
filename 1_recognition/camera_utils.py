
from dataclasses import dataclass
import logging
from pathlib import Path
import platform
import time
from typing import Optional, Any

from pygrabber.dshow_graph import FilterGraph

import cv2


logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    # --- HARDWARE & ROI ---
    # 0 is usually your laptop's built-in webcam. 
    # 1 or 2 will likely be the DroidCam Virtual Webcam. 
    # Change this number if it opens the wrong camera.
    camera_index: int = 1
    ip_address: str = "192.168.0.112"
    port: int = 4747

    frame_width: int = 640
    frame_height: int = 480

    roi_x_start: int = 150      # Adjust based on your setup
    roi_x_end: int = 950
    roi_y_start: int = 0
    roi_y_end: int = 350

    # --- ALGORITHM TUNING (Default Values) ---
    auto_exposure: bool = True  # Let camera handle exposure

    clahe_clip: float = 3.0     # Contrast enhancement (0.1 to 10.0)
    threshold: int = 25         # Sensitivity (0 to 255)
    kernel_width: int = 25      # Line length filter (odd numbers only)
    kernel_height: int = 5
    gap_fill_kernel: int = 35    # NEW: How wide a gap to bridge (in pixels)
    min_area: int = 20        # Keep even SMALL chunks (reflections)
    buffer_ratio: float = 0.5

    # --- RECONNECTION ---
    reconnect_max_retries: int = 5
    reconnect_delay: float = 1.0   # seconds between attempts



class CameraProcessing:
    def __init__(self, config: Optional[CameraConfig] = None,
                  config_file: Path = Path(),
                  config_key: str = 'Software.Camera') -> None:
        
        
        self.config_file = config_file
        # Load config from provided object or file, fallback to defaults
        if config is not None:
            self.config: CameraConfig = config
        else:
            self.config = CameraConfig()


    def setup_camera(self, cam_type: str = "webcam") -> cv2.VideoCapture:
        """Initialize and configure the camera. Returns an open VideoCapture."""
        self._cam_type = cam_type

        if cam_type == "webcam":
            api_preference = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.config.camera_index, api_preference)
        elif cam_type == "iphone_wifi":
            cap = self.setup_camera_wifi(self.config.ip_address, self.config.port)

        else:
            logger.error("Unsupported camera type: %s", self.config.cam_type)
            raise RuntimeError("Unsupported camera type.")

        # Best-effort settings (not all devices support these)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            logger.debug("Camera properties set: width=%s, height=%s", self.config.frame_width, self.config.frame_height)
            if not self.config.auto_exposure:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap.set(cv2.CAP_PROP_EXPOSURE, -5.0)
                cap.set(cv2.CAP_PROP_FOCUS, 0)
        except Exception:
            logger.debug("Some camera properties could not be set on this device.")

        if not cap.isOpened():
            logger.error("Could not open video device (index=%s)", self.config.camera_index)
            raise RuntimeError("Could not open video device.")

        logger.info("Camera opened (index=%s)", self.config.camera_index)
        
        return cap
    

    def setup_camera_wifi(self, ip_address: str, port: int) -> cv2.VideoCapture:
        """Initialize a VideoCapture for an iPhone camera stream over Wi-Fi."""
        cap = cv2.VideoCapture(f"http://{ip_address}:{port}/video")
        if not cap.isOpened():
            logger.error("Could not open iPhone camera stream at %s:%s", ip_address, port)
            raise RuntimeError("Could not open iPhone camera stream.")
        
        logger.info("iPhone camera stream opened at %s:%s", ip_address, port)
        return cap


    def read_frame(self, cap: cv2.VideoCapture) -> tuple[bool, Optional[Any], cv2.VideoCapture]:
        """Read a frame and force it to the configured frame size.

        Needed because network streams (e.g. DroidCam over Wi-Fi) ignore
        CAP_PROP_FRAME_WIDTH/HEIGHT, since the resolution is fixed by the
        sender before OpenCV ever sees the stream.

        On a failed read, attempts to reconnect (re-run setup_camera with the
        same cam_type) up to config.reconnect_max_retries times before giving
        up, since Wi-Fi streams routinely drop out. Returns the VideoCapture
        currently in use, since a reconnect replaces it with a new object.
        """
        ret, frame = cap.read()
        if not ret:
            cap = self._reconnect(cap)
            if cap is None:
                return False, None, cap
            ret, frame = cap.read()
            if not ret:
                return ret, frame, cap

        if frame.shape[1] != self.config.frame_width or frame.shape[0] != self.config.frame_height:
            frame = cv2.resize(frame, (self.config.frame_width, self.config.frame_height))

        return ret, frame, cap

    def _reconnect(self, cap: cv2.VideoCapture) -> Optional[cv2.VideoCapture]:
        """Release a dead capture and retry opening it, with backoff."""
        cap.release()
        cam_type = getattr(self, "_cam_type", "webcam")

        for attempt in range(1, self.config.reconnect_max_retries + 1):
            logger.warning(
                "Camera read failed, reconnecting (%s/%s)...",
                attempt, self.config.reconnect_max_retries,
            )
            time.sleep(self.config.reconnect_delay)
            try:
                return self.setup_camera(cam_type)
            except RuntimeError:
                continue

        logger.error("Giving up reconnecting after %s attempts.", self.config.reconnect_max_retries)
        return None


    """ 
        Camera testing functions (for development/debugging)
    """
    @staticmethod
    def iphone_camera_connection(ip_address: str, port: int):
        cap = cv2.VideoCapture(f"http://{ip_address}:{port}/video")

        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from iPhone camera stream.")
                break
            # Process the frame (e.g., display or save)
            cv2.imshow('iPhone Camera Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def webcam_connection(index: int = 1):
        # 0 is usually your laptop's built-in webcam. 
        # 1 or 2 will likely be the DroidCam Virtual Webcam. 
        # Change this number if it opens the wrong camera.
        # api_preference = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame from webcam stream.")
                break
                
            cv2.imshow('Webcam Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def list_cameras() -> list[dict[str, Any]]:
        """Return available DirectShow camera devices and whether each can produce frames."""
        try:
            devices = FilterGraph().get_input_devices()  # index order for DirectShow
        except Exception as exc:
            logger.exception("Failed to enumerate camera devices: %s", exc)
            return []

        cams = []
        for i, name in enumerate(devices):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            try:
                ok, _ = cap.read()
            except Exception:
                ok = False
            finally:
                cap.release()
            cams.append({"index": i, "name": name, "working": ok})
            
        return cams



def yolo_test():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Realtime recognition requires torch and ultralytics to be installed."
        ) from exc

    # Example usage of the CameraProcessing class
    camera_config = CameraConfig(camera_index=1, ip_address="192.168.0.112", port=4747)
    camera_processor = CameraProcessing(config=camera_config)
    cap = camera_processor.setup_camera("iphone_wifi")  # or "webcam" or "iphone_wifi"

    yolo_model = YOLO(Path(__file__).parent / "yolo26n-pose.pt")
    while True:
        loop_start = time.perf_counter()

        ret, frame, cap = camera_processor.read_frame(cap)
        if not ret:
            logger.error("Failed to read frame from camera, giving up.")
            break

        results = yolo_model(frame, verbose=False)
        result = results[0]


        # Ultralytics' own breakdown (ms) of the inference call itself.
        inference_ms = result.speed["inference"]
        # Wall-clock time for the whole read -> inference -> plot pipeline.
        total_ms = (time.perf_counter() - loop_start) * 1000

        annotated_frame = result.plot()
        cv2.putText(
            annotated_frame,
            f"inference: {inference_ms:.1f} ms | total: {total_ms:.1f} ms",
            (10, annotated_frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        logger.debug("YOLO latency: inference=%.1fms total=%.1fms", inference_ms, total_ms)

        cv2.imshow('YOLO Pose - iPhone Wi-Fi Camera Stream', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    




if __name__ == "__main__":
    # Example usage
    # CameraProcessing().list_cameras()
    # CameraProcessing().iphone_usb_connection()
    # CameraProcessing().iphone_camera_connection(ip_address="192.168.0.112", port=4747)
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG for more verbosity
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Starting camera processing example...")

    yolo_test()  # Run YOLO test with iPhone Wi-Fi camera stream

    # camera_config = CameraConfig(camera_index=1, ip_address="192.168.0.112", port=4747)  # Adjust index as needed

    # camera_processor = CameraProcessing(config = camera_config)
    # cap = camera_processor.setup_camera("iphone_wifi")  # or "webcam" or "iphone_wifi"
    # while True:
    #     ret, frame, cap = camera_processor.read_frame(cap)
    #     if not ret:
    #         logger.error("Failed to read frame from camera.")
    #         break

    #     cv2.imshow('iPhone Wi-Fi Camera Stream', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
            
    # cap.release()
    # cv2.destroyAllWindows()
