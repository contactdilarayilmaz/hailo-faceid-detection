import os
import hailo_platform as hpf
import numpy as np
import cv2


class ReIDModel:
    def __init__(self, hef_path):
        self.hef_path = hef_path
        self.hef = None
        self.device = None
        self.network_group = None
        self.input_params = None
        self.output_params = None

    def _ensure_initialized(self):
        if self.device is not None:
            return
        self.hef = hpf.HEF(self.hef_path)
        params = hpf.VDevice.create_params()
        # Try to use Multi-Process Service with group aligned to detection pipeline
        params.multi_process_service = True
        # Use a scheduler compatible with MPS
        params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN
        group_env = os.environ.get("HAILORT_VDEVICE_GROUP_ID", "1")
        try:
            params.group_id = str(int(group_env))
        except Exception:
            params.group_id = "1"
        self.device = hpf.VDevice(params)

        cfg_params = hpf.ConfigureParams.create_from_hef(
            self.hef, hpf.HailoStreamInterface.PCIe
        )
        self.network_group = self.device.configure(self.hef, cfg_params)[0]
        self.input_params = hpf.InputVStreamParams.make_from_network_group(self.network_group)
        self.output_params = hpf.OutputVStreamParams.make_from_network_group(self.network_group)

    def infer(self, img):
        self._ensure_initialized()
        img_resized = cv2.resize(img, (128, 256)) / 255.0
        img_resized = np.transpose(img_resized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        
        with hpf.InferVStreams(self.network_group, self.input_params, self.output_params) as infer_vstreams:
            
            try:
                input_name = next(iter(self.input_params))
            except TypeError:
                input_name = next(iter(self.input_params.get()))
            outputs = infer_vstreams.infer({input_name: img_resized})
        return list(outputs.values())[0].flatten()



class ReIDPipeline:
    def __init__(self):
        self.reid = ReIDModel("/home/pi/hailo-rpi5-examples/resources/models/hailo8/repvgg_reid.hef")
        self.min_crop_area = 1600
        self.debug_reports = 0
        try:
            self.debug_report_limit = int(os.environ.get("REID_DEBUG_LIMIT", "20"))
        except Exception:
            self.debug_report_limit = 20

    @staticmethod
    def _get_attr(obj, names):
        for name in names:
            if hasattr(obj, name):
                value = getattr(obj, name)
                try:
                    return value() if callable(value) else value
                except Exception:
                    continue
        return None

    def _extract_bbox_and_conf(self, det):
        confidence = None
        source = det
        if isinstance(det, dict):
            confidence = det.get("confidence")
            source = det.get("bbox", det)

        if isinstance(source, (list, tuple)) and len(source) == 4:
            try:
                bbox = tuple(float(v) for v in source)
            except (TypeError, ValueError):
                return None, confidence
        else:
            x1 = self._get_attr(source, ["get_xmin", "get_left", "xmin", "x_min", "left"])
            y1 = self._get_attr(source, ["get_ymin", "get_top", "ymin", "y_min", "top"])
            x2 = self._get_attr(source, ["get_xmax", "get_right", "xmax", "x_max", "right"])
            y2 = self._get_attr(source, ["get_ymax", "get_bottom", "ymax", "y_max", "bottom"])
            if None in (x1, y1, x2, y2):
                return None, confidence
            try:
                bbox = (float(x1), float(y1), float(x2), float(y2))
            except (TypeError, ValueError):
                return None, confidence

        if confidence is None and source is not None:
            confidence = self._get_attr(source, ["get_confidence", "confidence", "score"])

        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None

        return bbox, confidence

    @staticmethod
    def _estimate_quality(bbox, frame_area, confidence):
        x1, y1, x2, y2 = bbox
        det_area = max(1.0, float((x2 - x1) * (y2 - y1)))
        norm_area = det_area / max(1.0, float(frame_area))
        area_quality = float(np.clip(norm_area * 4.0, 0.1, 1.0))
        if confidence is not None:
            confidence_quality = float(np.clip(confidence, 0.1, 1.0))
            combined = area_quality * confidence_quality
            return float(np.clip(combined, 0.1, 1.0))
        return area_quality

    def _debug_skip(self, det_index, reason, **kwargs):
        if self.debug_reports >= self.debug_report_limit:
            return
        details = ", ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        message = f"[reid-debug] det[{det_index}] skipped: {reason}"
        if details:
            message += f" ({details})"
        print(message)
        self.debug_reports += 1

    def get_embeddings(self, frame, detections):
        results = []
        if frame is None or detections is None:
            return results

        frame_height, frame_width = frame.shape[:2]
        frame_area = max(1, frame_height * frame_width)

        for det_index, det in enumerate(detections):
            bbox_conf = self._extract_bbox_and_conf(det)
            if bbox_conf is None:
                self._debug_skip(det_index, "bbox not resolved")
                continue
            bbox, confidence = bbox_conf
            if bbox is None:
                self._debug_skip(det_index, "bbox None after extraction")
                continue
            x1, y1, x2, y2 = bbox

            # Handle normalized coordinates (0-1 range)
            if (
                frame_width > 1
                and frame_height > 1
                and 0.0 <= x1 <= 1.2
                and 0.0 <= x2 <= 1.2
                and 0.0 <= y1 <= 1.2
                and 0.0 <= y2 <= 1.2
            ):
                x1 *= frame_width
                x2 *= frame_width
                y1 *= frame_height
                y2 *= frame_height

            x1 = int(np.clip(x1, 0, frame_width))
            y1 = int(np.clip(y1, 0, frame_height))
            x2 = int(np.clip(x2, 0, frame_width))
            y2 = int(np.clip(y2, 0, frame_height))

            if x2 <= x1 or y2 <= y1:
                self._debug_skip(
                    det_index,
                    "invalid bbox after clipping",
                    raw_bbox=bbox,
                    clipped=(x1, y1, x2, y2),
                )
                continue

            bw = float(x2 - x1)
            bh = float(y2 - y1)
            if bw < 2.0 or bh < 4.0:
                self._debug_skip(det_index, "bbox too small", width=bw, height=bh)
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            bw *= 1.10
            bh *= 1.10
            target_ratio = 0.5  # width:height = 1:2
            cur_ratio = bw / max(bh, 1e-6)
            if cur_ratio > target_ratio:
                bh = bw / target_ratio
            else:
                bw = bh * target_ratio

            nx1 = int(max(0, min(frame_width, cx - bw / 2.0)))
            ny1 = int(max(0, min(frame_height, cy - bh / 2.0)))
            nx2 = int(max(0, min(frame_width, cx + bw / 2.0)))
            ny2 = int(max(0, min(frame_height, cy + bh / 2.0)))

            if nx2 <= nx1 or ny2 <= ny1:
                self._debug_skip(
                    det_index,
                    "adjusted crop invalid",
                    crop_bbox=(nx1, ny1, nx2, ny2),
                )
                continue

            crop_area = (nx2 - nx1) * (ny2 - ny1)
            if crop_area < self.min_crop_area:
                self._debug_skip(det_index, "crop too small", crop_area=crop_area)
                continue

            crop = frame[ny1:ny2, nx1:nx2]
            if crop.size == 0:
                self._debug_skip(det_index, "empty crop array")
                continue

            embedding = self.reid.infer(crop)
            quality = self._estimate_quality((x1, y1, x2, y2), frame_area, confidence)

            results.append(
                {
                    "det_index": det_index,
                    "embedding": embedding,
                    "bbox": (x1, y1, x2, y2),
                    "crop_bbox": (nx1, ny1, nx2, ny2),
                    "confidence": confidence,
                    "quality": quality,
                }
            )

        return results

if __name__ == "__main__":
    print("ReID pipeline loaded successfully.")
    

