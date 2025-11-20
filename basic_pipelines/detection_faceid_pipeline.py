import os
import hailo_platform as hpf
import numpy as np
import cv2


# FaceEmbeddingModel: thin helper around the ArcFace HEF to produce embeddings
class FaceEmbeddingModel:
    def __init__(self, hef_path):
        self.hef_path = hef_path
        self.hef = None
        self.device = None
        self.network_group = None
        self.input_params = None
        self.output_params = None
        self.input_name = None
        self.output_name = None
        self.input_shape = (3, 112, 112)

    def _resolve_stream_name(self, params):
        try:
            return next(iter(params))
        except TypeError:
            return next(iter(params.get()))

    def _ensure_initialized(self):
        if self.device is not None:
            return

        self.hef = hpf.HEF(self.hef_path)

        params = hpf.VDevice.create_params()
        params.multi_process_service = True
        params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN
        group_env = os.environ.get("HAILORT_VDEVICE_GROUP_ID", "1")
        try:
            params.group_id = str(int(group_env))
        except Exception:
            params.group_id = "1"
        self.device = hpf.VDevice(params)

        cfg_params = hpf.ConfigureParams.create_from_hef(self.hef, hpf.HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(self.hef, cfg_params)[0]
        self.input_params = hpf.InputVStreamParams.make_from_network_group(self.network_group)
        self.output_params = hpf.OutputVStreamParams.make_from_network_group(self.network_group)

        self.input_name = self._resolve_stream_name(self.input_params)
        self.output_name = self._resolve_stream_name(self.output_params)

        try:
            infos = self.hef.get_input_vstream_infos()
            if infos:
                first_info = next(iter(infos.values()))
                shape = getattr(first_info, "shape", None)
                if shape and len(shape) >= 3:
                    self.input_shape = tuple(int(v) for v in shape)
        except Exception:
            pass

    def _preprocess(self, face_img):
        _, height, width = self.input_shape
        face_resized = cv2.resize(face_img, (width, height))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        # Model expects UINT8 in NHWC format (batch, Height, Width, Channels)
        # Add batch dimension: (112, 112, 3) -> (1, 112, 112, 3)
        return face_rgb.astype(np.uint8)[np.newaxis, ...]

    def infer(self, face_img):
        if face_img is None or face_img.size == 0:
            return None

        self._ensure_initialized()
        input_tensor = self._preprocess(face_img)

        with hpf.InferVStreams(self.network_group, self.input_params, self.output_params) as infer_vstreams:
            outputs = infer_vstreams.infer({self.input_name: input_tensor})

        if not outputs:
            return None

        embedding = np.array(list(outputs.values())[0]).flatten()
        norm = np.linalg.norm(embedding)
        if norm > 1e-6 and np.isfinite(norm):
            embedding = embedding / norm
        
        return embedding


# FaceIDPipeline: extracts face crops from person detections and runs ArcFace
class FaceIDPipeline:
    def __init__(self):
        self.model = FaceEmbeddingModel(
            "/home/pi/hailo-rpi5-examples/resources/models/hailo8/arcface_mobilefacenet.hef"
        )
        self.min_face_crop = 4000
        self.debug_reports = 0
        try:
            self.debug_report_limit = int(os.environ.get("FACE_DEBUG_LIMIT", "20"))
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
        area_quality = float(np.clip(norm_area * 6.0, 0.1, 1.0))
        if confidence is not None:
            confidence_quality = float(np.clip(confidence, 0.1, 1.0))
            combined = area_quality * confidence_quality
            return float(np.clip(combined, 0.1, 1.0))
        return area_quality

    def _debug_skip(self, det_index, reason, **kwargs):
        if self.debug_reports >= self.debug_report_limit:
            return
        details = ", ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        message = f"[face-id-debug] det[{det_index}] skipped: {reason}"
        if details:
            message += f" ({details})"
        print(message)
        self.debug_reports += 1

    def _compute_face_crop(self, frame, bbox):
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return None

        # More conservative face crop: larger and more centered for better embeddings
        face_height = height * 0.55
        face_width = width * 0.65
        center_x = x1 + width * 0.5
        center_y = y1 + height * 0.25  # Slightly higher for better face centering

        fx1 = int(np.clip(center_x - face_width / 2.0, 0, frame_width))
        fx2 = int(np.clip(center_x + face_width / 2.0, 0, frame_width))
        fy1 = int(np.clip(center_y - face_height / 2.0, 0, frame_height))
        fy2 = int(np.clip(center_y + face_height / 2.0, 0, frame_height))

        if fx2 <= fx1 or fy2 <= fy1:
            return None

        return (fx1, fy1, fx2, fy2)

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

            face_bbox = self._compute_face_crop(frame, (x1, y1, x2, y2))
            if face_bbox is None:
                self._debug_skip(det_index, "face crop heuristic failed")
                continue

            fx1, fy1, fx2, fy2 = face_bbox
            face_width = fx2 - fx1
            face_height = fy2 - fy1
            face_area = face_width * face_height
            # Check both area and minimum dimensions (70x70 minimum for meaningful embeddings)
            if face_area < self.min_face_crop or face_width < 70 or face_height < 70:
                self._debug_skip(det_index, "face crop too small", area=face_area, width=face_width, height=face_height)
                continue

            face_crop = frame[fy1:fy2, fx1:fx2]
            if face_crop.size == 0:
                self._debug_skip(det_index, "empty face crop")
                continue

            embedding = self.model.infer(face_crop)
            if embedding is None:
                self._debug_skip(det_index, "embedding failed")
                continue

            # Debug: log embedding and face crop info
            emb_norm = np.linalg.norm(embedding)
            emb_first5 = embedding[:5] if len(embedding) >= 5 else embedding
            print(f"[embedding-debug] det[{det_index}] face_bbox={face_bbox} emb_norm={emb_norm:.6f} first5={emb_first5}")

            quality = self._estimate_quality(face_bbox, frame_area, confidence)

            results.append(
                {
                    "det_index": det_index,
                    "embedding": embedding,
                    "bbox": (x1, y1, x2, y2),
                    "face_bbox": face_bbox,
                    "confidence": confidence,
                    "quality": quality,
                }
            )

        return results


if __name__ == "__main__":
    print("FaceID pipeline loaded successfully.")
