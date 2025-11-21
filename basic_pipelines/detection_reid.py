from pathlib import Path
import os
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import cv2
import hailo
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class

from basic_pipelines.detection_reid_pipeline import ReIDPipeline
from basic_pipelines.utils import Tracker

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.reid = ReIDPipeline()
        self.tracker = Tracker()
        # FPS calculation
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    try:
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    except Exception as e:
        
        return Gst.PadProbeReturn.OK

    detection_records = []
    for det in detections:
        try:
            label = det.get_label()
        except Exception:
            continue
        if label != "person":
            continue
        bbox = det.get_bbox()
        confidence = None
        if hasattr(det, "get_confidence"):
            try:
                confidence = det.get_confidence()
            except Exception:
                confidence = None
        detection_records.append({"bbox": bbox, "confidence": confidence})

    def _buffer_to_numpy(pad, buffer):
        caps = pad.get_current_caps()
        if not caps or caps.get_size() == 0:
            return None
        s = caps.get_structure(0)
        fmt = s.get_string("format")
        width = s.get_value("width")
        height = s.get_value("height")

        success, mapinfo = buffer.map(Gst.MapFlags.READ)
        if not success:
            return None
        try:
            data = np.frombuffer(mapinfo.data, dtype=np.uint8)
            if fmt in ("RGBA", "BGRA"):
                arr = data.reshape((height, width, 4))
                if fmt == "RGBA":
                    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                else:
                    return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            if fmt in ("RGB",):
                arr = data.reshape((height, width, 3))
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            if fmt in ("BGR",):
                return data.reshape((height, width, 3))
            if fmt in ("BGRx", "xBGR"):
                arr = data.reshape((height, width, 4))
                return arr[:, :, :3]
            if fmt == "NV12":
                # NV12 is Y plane followed by interleaved UV plane
                y_size = width * height
                if data.size < int(y_size * 3 / 2):
                    return None
                yuv = data.reshape((int(height * 3 / 2), width))
                return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        finally:
            buffer.unmap(mapinfo)
        return None

    # Calculate FPS (always, for every frame)
    user_data.fps_frame_count += 1
    current_time = time.time()
    elapsed = current_time - user_data.last_fps_time
    if elapsed >= 1.0:  # Update FPS every second
        user_data.current_fps = user_data.fps_frame_count / elapsed
        user_data.fps_frame_count = 0
        user_data.last_fps_time = current_time
    
    detection_count = len(detection_records)
    frame_num = user_data.get_count()

    frame = None
    if detection_records:
        frame = _buffer_to_numpy(pad, buffer)

    embeddings = []
    if frame is not None and detection_records:
        embeddings = user_data.reid.get_embeddings(frame, detection_records)
        if detection_records and not embeddings:
            print(
                f"[debug] frame {frame_num}: all {len(detection_records)} detections filtered before tracking"
            )

    person_ids = user_data.tracker.assign(embeddings, frame_num, detection_count)

    assigned_ids = [pid for pid in person_ids if pid is not None]

    # Print formatted output: Frame | FPS | People detected | IDs
    ids_str = ", ".join(map(str, assigned_ids)) if assigned_ids else "None"
    fps_str = f"{user_data.current_fps:.1f}" if user_data.current_fps > 0 else "0.0"
    print(f"Frame {frame_num:5d} | FPS: {fps_str:5s} | People detected: {detection_count:2d} | IDs: [{ids_str}]")

    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)
    # Force-select the available Hailo device to avoid "no free devices" selection issues
    os.environ.setdefault("HAILO_DEVICE_IDS", "0001:01:00.0")
    # Set MPS group ID to ensure ReID pipeline shares the same VDevice group as detection pipeline
    os.environ.setdefault("HAILORT_VDEVICE_GROUP_ID", "1")
    # Disable problematic video sinks to avoid KMS permission errors
    os.environ.setdefault("GST_PLUGIN_FEATURE_RANK", "kmssink:NONE,glimagesink:NONE,waylandsink:NONE")

    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
