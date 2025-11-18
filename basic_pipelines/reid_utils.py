import sqlite3
import threading
import time

import numpy as np

try:
    from annoy import AnnoyIndex
except ImportError:  # pragma: no cover
    AnnoyIndex = None


# ReIDTracker: handles assigning IDs, persisting embeddings and optional ANN search
class ReIDTracker:
    def __init__(
        self,
        similarity_threshold=0.58,
        ema_beta=0.12,
        max_inactive_frames=900,
        max_prototypes_per_id=6,
        db_path=None,#"/home/pi/hailo-rpi5-examples/data/reid.sqlite",
        use_ann=0,
        ann_metric="angular",
        ann_trees=10,
        ann_candidates=20,
    ):
        self.similarity_threshold = similarity_threshold
        self.ema_beta = ema_beta
        self.max_inactive_frames = max_inactive_frames
        self.max_prototypes_per_id = max_prototypes_per_id
        self.gallery = []
        self.next_id = 1
        self.db_path = db_path
        self._db_lock = threading.Lock()
        if self.db_path:
            self._init_db()

        self.use_ann = bool(use_ann) and (AnnoyIndex is not None)
        if use_ann and AnnoyIndex is None:
            print("[reid-utils] Annoy package not found; disabling ANN mode.")
        self.ann_metric = ann_metric
        self.ann_trees = ann_trees
        self.ann_candidates = ann_candidates
        self._ann_index = None
        self._ann_dim = None
        self._ann_id_map = {}
        self._ann_dirty = True

    # initialize SQLite schema if persistence is enabled
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reid_entries (
                    id INTEGER PRIMARY KEY,
                    centroid BLOB,
                    last_seen INTEGER,
                    last_seen_ts INTEGER,
                    missed_frames INTEGER,
                    total_hits INTEGER,
                    quality REAL,
                    created_at INTEGER
                )
                """
            )
            conn.commit()
            columns = [
                row[1] for row in conn.execute("PRAGMA table_info(reid_entries);")
            ]
            if "last_seen_ts" not in columns:
                conn.execute("ALTER TABLE reid_entries ADD COLUMN last_seen_ts INTEGER")
                conn.commit()

    def _persist_entry(self, entry):
        if not self.db_path:
            return
        payload = (
            entry["id"],
            entry["centroid"].tobytes() if entry.get("centroid") is not None else None,
            entry.get("last_seen", 0),
            entry.get("last_seen_ts", entry.get("created_at", int(time.time()))),
            entry.get("missed_frames", 0),
            entry.get("hits", 0),
            entry.get("quality", 0.0),
            entry.get("created_at", int(time.time())),
        )
        with self._db_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO reid_entries (id, centroid, last_seen, last_seen_ts, missed_frames, total_hits, quality, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    centroid=excluded.centroid,
                    last_seen=excluded.last_seen,
                    last_seen_ts=excluded.last_seen_ts,
                    missed_frames=excluded.missed_frames,
                    total_hits=excluded.total_hits,
                    quality=excluded.quality
                """,
                payload,
            )
            conn.commit()

    def _delete_entry(self, entry_id):
        if not self.db_path:
            return
        with self._db_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM reid_entries WHERE id=?", (entry_id,))
            conn.commit()

    @staticmethod
    def _normalize(emb):
        if emb is None:
            return None
        emb = np.asarray(emb, dtype=np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm < 1e-6 or not np.isfinite(norm):
            return None
        return emb / norm

    def _similarity(self, emb, entry):
        prototypes = entry.get("prototypes") or [entry["centroid"]]
        sims = [float(np.dot(emb, proto)) for proto in prototypes]
        return max(sims) if sims else float(np.dot(emb, entry["centroid"]))

    def _update_entry(self, entry, emb, frame_idx, bbox, quality):
        entry["centroid"] = self._normalize(
            (1.0 - self.ema_beta) * entry["centroid"] + self.ema_beta * emb
        )
        entry["prototypes"].append(emb.copy())
        if len(entry["prototypes"]) > self.max_prototypes_per_id:
            entry["prototypes"].pop(0)
        entry["count"] += 1
        entry["last_seen"] = frame_idx
        entry["last_seen_ts"] = int(time.time())
        entry["bbox"] = bbox
        entry["quality"] = max(entry.get("quality", 0.0), float(quality))
        entry["missed_frames"] = 0
        entry["hits"] = entry.get("hits", 0) + 1
        self._ann_dirty = True

    def _create_entry(self, emb, frame_idx, bbox, quality):
        entry = {
            "id": self.next_id,
            "centroid": emb.copy(),
            "prototypes": [emb.copy()],
            "count": 1,
            "last_seen": frame_idx,
            "last_seen_ts": int(time.time()),
            "bbox": bbox,
            "quality": float(quality),
            "missed_frames": 0,
            "hits": 1,
            "created_at": int(time.time()),
        }
        self.next_id += 1
        self.gallery.append(entry)
        self._persist_entry(entry)
        self._ann_dirty = True
        return entry

    def _prune(self):
        self.gallery = [
            entry
            for entry in self.gallery
            if entry.get("missed_frames", 0) <= self.max_inactive_frames
        ]
        self._ann_dirty = True

    def _build_ann_index(self):
        if not self.use_ann or not self.gallery:
            self._ann_index = None
            self._ann_id_map = {}
            self._ann_dim = None
            self._ann_dirty = False
            return

        dim = len(self.gallery[0]["centroid"])
        ann = AnnoyIndex(dim, metric=self.ann_metric)
        id_map = {}
        idx = 0
        for entry in self.gallery:
            centroid = entry.get("centroid")
            if centroid is None:
                continue
            ann.add_item(idx, centroid)
            id_map[idx] = entry
            idx += 1

        if idx == 0:
            self._ann_index = None
            self._ann_id_map = {}
            self._ann_dim = None
            self._ann_dirty = False
            return

        ann.build(self.ann_trees)
        self._ann_index = ann
        self._ann_dim = dim
        self._ann_id_map = id_map
        self._ann_dirty = False

    def _get_ann_candidates(self, vector):
        if not self.use_ann:
            return self.gallery
        if self._ann_dirty or self._ann_index is None:
            self._build_ann_index()
        if self._ann_index is None:
            return self.gallery
        k = max(int(self.ann_candidates), 1)
        idxs = self._ann_index.get_nns_by_vector(vector, k, include_distances=False)
        candidates = []
        for idx in idxs:
            entry = self._ann_id_map.get(idx)
            if entry is not None:
                candidates.append(entry)
        return candidates or self.gallery

    # main entry: match incoming detections to stored embeddings
    def assign(self, detections_info, frame_idx, total_detections):
        assignment = [None] * total_detections
        if detections_info is None:
            detections_info = []

        processed = []
        for det_idx, det in enumerate(detections_info):
            det_index = det.get("det_index", det_idx)
            emb = self._normalize(det.get("embedding"))
            if emb is None:
                continue
            processed.append(
                {
                    "det_index": det_index,
                    "embedding": emb,
                    "bbox": det.get("bbox"),
                    "quality": float(det.get("quality", 1.0) or 1.0),
                }
            )

        for entry in self.gallery:
            entry["missed_frames"] = entry.get("missed_frames", 0) + 1

        if not processed:
            self._prune()
            return assignment

        used_ids = set()

        for det in processed:
            det_index = det["det_index"]
            if det_index is None or det_index >= total_detections:
                continue

            best_entry = None
            best_sim = -1.0
            candidate_entries = self._get_ann_candidates(det["embedding"])
            for entry in candidate_entries:
                if entry["id"] in used_ids:
                    continue
                sim = self._similarity(det["embedding"], entry)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_entry is not None and best_sim >= self.similarity_threshold:
                self._update_entry(best_entry, det["embedding"], frame_idx, det.get("bbox"), det.get("quality", 1.0))
                self._persist_entry(best_entry)
                assignment[det_index] = best_entry["id"]
                used_ids.add(best_entry["id"])
            else:
                new_entry = self._create_entry(
                    det["embedding"],
                    frame_idx,
                    det.get("bbox"),
                    det.get("quality", 1.0),
                )
                assignment[det_index] = new_entry["id"]
                used_ids.add(new_entry["id"])

        self._prune()
        return assignment
