import atexit
import sqlite3
import threading
import time
from queue import Empty, Full, Queue

import numpy as np

try:
    from annoy import AnnoyIndex
except ImportError:  # pragma: no cover
    AnnoyIndex = None


# Tracker: handles assigning IDs, persisting embeddings and optional ANN search
class Tracker:
    def __init__(
        self,
        similarity_threshold=0.58,
        ema_beta=0.12,
        max_inactive_frames=900,
        max_prototypes_per_id=6,
        db_path=None,
        use_ann=0,
        ann_metric="angular",
        ann_trees=10,
        ann_candidates=20,
        db_recall_limit=0,
    ):
        self.similarity_threshold = similarity_threshold
        self.ema_beta = ema_beta
        self.max_inactive_frames = max_inactive_frames
        self.max_prototypes_per_id = max_prototypes_per_id
        self.gallery = []  # Active gallery (pruned periodically)
        self.db_gallery = []  # Persistent gallery buffer (never pruned, loaded from DB)
        self.next_id = 1
        
        self.db_path = db_path
        self._db_queue = None
        self._db_thread = None
        self._db_stop = None
        self._db_queue_dropped = False
        
        if self.db_path:
            self._init_db()
            self._start_db_worker()

        self.use_ann = bool(use_ann) and (AnnoyIndex is not None)
        if use_ann and AnnoyIndex is None:
            print("[tracker] Annoy package not found; disabling ANN mode.")
        self.ann_metric = ann_metric
        self.ann_trees = ann_trees
        self.ann_candidates = ann_candidates
        self._ann_index = None
        self._ann_dim = None
        self._ann_id_map = {}
        self._ann_dirty = True
        self._last_ann_rebuild_frame = 0
        self._ann_rebuild_lock = threading.Lock()
        self._ann_rebuild_in_progress = False
        try:
            import os
            self._ann_rebuild_interval = int(os.environ.get("REID_ANN_REBUILD_INTERVAL", "30"))
        except (ValueError, TypeError):
            self._ann_rebuild_interval = 30
        
        self.db_recall_limit = max(int(db_recall_limit or 0), 0)
        self._load_state_from_db()

    # ========== Database Operations ==========
    
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
            columns = [row[1] for row in conn.execute("PRAGMA table_info(reid_entries);")]
            if "last_seen_ts" not in columns:
                conn.execute("ALTER TABLE reid_entries ADD COLUMN last_seen_ts INTEGER")
                conn.commit()

    def _load_state_from_db(self):
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                max_id_row = cursor.execute("SELECT MAX(id) FROM reid_entries").fetchone()
                if max_id_row and max_id_row[0] is not None:
                    self.next_id = max(self.next_id, int(max_id_row[0]) + 1)

                if self.db_recall_limit <= 0:
                    return

                rows = cursor.execute(
                    """
                    SELECT id, centroid, last_seen, last_seen_ts, missed_frames,
                           total_hits, quality, created_at
                    FROM reid_entries
                    WHERE centroid IS NOT NULL
                    ORDER BY last_seen_ts DESC
                    LIMIT ?
                    """,
                    (self.db_recall_limit,),
                ).fetchall()
        except sqlite3.Error as exc:
            print(f"[tracker] Warning: failed to load DB state: {exc}")
            return

        restored = []
        for rid, blob, last_seen, last_seen_ts, missed_frames, total_hits, quality, created_at in rows:
            if missed_frames is not None and missed_frames > self.max_inactive_frames:
                continue
            if not blob:
                continue
            try:
                centroid = np.frombuffer(blob, dtype=np.float32).copy()
            except ValueError:
                continue
            centroid = self._normalize(centroid)
            if centroid is None:
                continue
            entry = {
                "id": rid,
                "centroid": centroid,
                "prototypes": [centroid.copy()],
                "count": total_hits or 0,
                "last_seen": last_seen or 0,
                "last_seen_ts": last_seen_ts or created_at or int(time.time()),
                "bbox": None,
                "quality": float(quality or 0.0),
                "missed_frames": missed_frames or 0,
                "hits": total_hits or 0,
                "created_at": created_at or int(time.time()),
            }
            restored.append(entry)

        if restored:
            self.gallery.extend(restored)
            self.db_gallery.extend(restored)
            self._ann_dirty = True

    def _start_db_worker(self):
        if self._db_queue is not None:
            return
        self._db_queue = Queue(maxsize=4096)
        self._db_stop = threading.Event()
        self._db_thread = threading.Thread(
            target=self._db_worker,
            name="TrackerDBWriter",
            daemon=True,
        )
        self._db_thread.start()
        atexit.register(self._shutdown_db_worker)

    def _shutdown_db_worker(self):
        if not self._db_queue:
            return
        if self._db_stop:
            self._db_stop.set()
        try:
            self._db_queue.put_nowait(("STOP", None))
        except Full:
            pass
        if self._db_thread and self._db_thread.is_alive():
            self._db_thread.join(timeout=1.0)
        self._db_queue = None

    def _db_worker(self):
        """Background thread that processes DB operations from queue."""
        if not self.db_path:
            return
        
        upsert_sql = """
            INSERT INTO reid_entries (id, centroid, last_seen, last_seen_ts, missed_frames, total_hits, quality, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                centroid=excluded.centroid,
                last_seen=excluded.last_seen,
                last_seen_ts=excluded.last_seen_ts,
                missed_frames=excluded.missed_frames,
                total_hits=excluded.total_hits,
                quality=excluded.quality
        """
        delete_sql = "DELETE FROM reid_entries WHERE id=?"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                while True:
                    # Check if we should stop
                    if self._db_stop and self._db_stop.is_set() and self._db_queue.empty():
                        break
                    
                    # Get operation from queue (wait max 0.5 seconds)
                    try:
                        op, payload = self._db_queue.get(timeout=0.5)
                    except Empty:
                        continue
                    
                    # Process operation (task_done will be called in finally block)
                    try:
                        # Handle STOP signal
                        if op == "STOP":
                            break
                        
                        # Execute DB operation
                        if op == "UPSERT":
                            conn.execute(upsert_sql, payload)
                            conn.commit()
                        elif op == "DELETE":
                            conn.execute(delete_sql, (payload,))
                            conn.commit()
                        else:
                            print(f"[tracker] Unknown DB operation: {op}")
                            
                    except sqlite3.Error as exc:
                        print(f"[tracker] DB worker error: {exc}")
                    finally:
                        # Always mark task as done, even if operation failed
                        self._db_queue.task_done()
                        
        except sqlite3.Error as exc:
            print(f"[tracker] DB worker aborted: {exc}")

    def _enqueue_db_op(self, op, payload):
        if not self._db_queue:
            return
        try:
            self._db_queue.put_nowait((op, payload))
        except Full:
            if not self._db_queue_dropped:
                print("[tracker] Warning: DB queue full, dropping writes.")
                self._db_queue_dropped = True

    def _persist_entry(self, entry):
        if not self.db_path:
            return
        centroid = entry.get("centroid")
        payload = (
            entry["id"],
            centroid.tobytes() if centroid is not None else None,
            entry.get("last_seen", 0),
            entry.get("last_seen_ts", entry.get("created_at", int(time.time()))),
            entry.get("missed_frames", 0),
            entry.get("hits", 0),
            entry.get("quality", 0.0),
            entry.get("created_at", int(time.time())),
        )
        self._enqueue_db_op("UPSERT", payload)

    # ========== Embedding Operations ==========

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

    # ========== Entry Management ==========

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
        # Add copy to persistent buffer
        db_entry = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in entry.items()}
        db_entry["prototypes"] = [entry["centroid"].copy()]
        self.db_gallery.append(db_entry)
        self._persist_entry(entry)
        self._ann_dirty = True
        return entry

    def _prune(self):
        self.gallery = [
            entry for entry in self.gallery
            if entry.get("missed_frames", 0) <= self.max_inactive_frames
        ]
        self._ann_dirty = True

    # ========== ANN Index Operations ==========

    def _build_ann_index(self):
        all_entries = self.gallery + self.db_gallery
        if not self.use_ann or not all_entries:
            self._ann_index = None
            self._ann_id_map = {}
            self._ann_dim = None
            self._ann_dirty = False
            return

        dim = len(all_entries[0]["centroid"])
        ann = AnnoyIndex(dim, metric=self.ann_metric)
        id_map = {}
        idx = 0
        seen_ids = set()
        for entry in all_entries:
            entry_id = entry.get("id")
            if entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
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
        # Atomically update index (thread-safe)
        with self._ann_rebuild_lock:
            self._ann_index = ann
            self._ann_dim = dim
            self._ann_id_map = id_map
            self._ann_dirty = False

    def _build_ann_index_async(self):
        """Build ANN index asynchronously in background thread (non-blocking)."""
        if self._ann_rebuild_in_progress:
            return
        
        self._ann_rebuild_in_progress = True
        
        def rebuild_worker():
            try:
                self._build_ann_index()
            finally:
                self._ann_rebuild_in_progress = False
        
        thread = threading.Thread(target=rebuild_worker, daemon=True, name="ANNRebuild")
        thread.start()

    def _get_ann_candidates(self, vector, frame_idx=None):
        if not self.use_ann:
            return self.gallery + self.db_gallery
        
        # Check if rebuild is needed
        should_rebuild = False
        if self._ann_index is None:
            should_rebuild = True
        elif self._ann_dirty:
            if frame_idx is not None:
                frames_since_rebuild = frame_idx - self._last_ann_rebuild_frame
                if frames_since_rebuild >= self._ann_rebuild_interval:
                    should_rebuild = True
            else:
                should_rebuild = True
        
        # Use existing index if available (thread-safe read)
        with self._ann_rebuild_lock:
            ann_index = self._ann_index
            ann_id_map = self._ann_id_map
        
        if ann_index is not None:
            k = max(int(self.ann_candidates), 1)
            idxs = ann_index.get_nns_by_vector(vector, k, include_distances=False)
            candidates = [ann_id_map.get(idx) for idx in idxs if ann_id_map.get(idx) is not None]
            if candidates:
                # Trigger async rebuild if needed (non-blocking)
                if should_rebuild and not self._ann_rebuild_in_progress:
                    self._build_ann_index_async()
                    if frame_idx is not None:
                        self._last_ann_rebuild_frame = frame_idx
                return candidates
        
        # No index available - must rebuild synchronously (first time only)
        if should_rebuild:
            self._build_ann_index()
            if frame_idx is not None:
                self._last_ann_rebuild_frame = frame_idx
            
            # Try again with new index
            with self._ann_rebuild_lock:
                ann_index = self._ann_index
                ann_id_map = self._ann_id_map
            
            if ann_index is not None:
                k = max(int(self.ann_candidates), 1)
                idxs = ann_index.get_nns_by_vector(vector, k, include_distances=False)
                candidates = [ann_id_map.get(idx) for idx in idxs if ann_id_map.get(idx) is not None]
                if candidates:
                    return candidates
        
        # Fallback to linear search
        return self.gallery + self.db_gallery

    def _get_db_gallery_from_ann(self):
        """Get db_gallery entries from ANN index if available."""
        if not self.use_ann or self._ann_id_map is None:
            return self.db_gallery
        gallery_ids = {entry["id"] for entry in self.gallery}
        db_entries = [entry for entry in self._ann_id_map.values() if entry.get("id") not in gallery_ids]
        return db_entries if db_entries else self.db_gallery

    # ========== Search Operations ==========

    def _search_in_entries(self, embedding, entries, used_ids, threshold=None):
        """Search for best matching entry in given entries list."""
        best_entry = None
        best_sim = -1.0
        threshold = threshold or self.similarity_threshold
        
        for entry in entries:
            if entry["id"] in used_ids:
                continue
            sim = self._similarity(embedding, entry)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
        
        if best_entry is not None and best_sim >= threshold:
            return best_entry, best_sim
        return None, best_sim

    def _revive_from_db(self, embedding, used_ids, frame_idx=None):
        """Search in db_gallery for a matching entry."""
        if embedding is None:
            return None, -1.0

        gallery_ids = {entry["id"] for entry in self.gallery}
        best_entry = None
        best_sim = -1.0

        if self.use_ann:
            # Get expanded candidates from ANN
            expanded_candidates = max(int(self.ann_candidates * 3), 50)
            if self._ann_index is None:
                candidate_entries = self.db_gallery
            else:
                k = max(expanded_candidates, 1)
                idxs = self._ann_index.get_nns_by_vector(embedding, k, include_distances=False)
                candidate_entries = [self._ann_id_map.get(idx) for idx in idxs if self._ann_id_map.get(idx) is not None]
                if not candidate_entries:
                    candidate_entries = self.db_gallery
            
            # Filter to db_gallery entries only
            for entry in candidate_entries:
                entry_id = entry.get("id")
                if entry_id in gallery_ids or entry_id in used_ids:
                    continue
                sim = self._similarity(embedding, entry)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry
            
            # Linear search fallback if needed
            if best_entry is None or (best_sim < (self.similarity_threshold - 0.02) and best_sim > (self.similarity_threshold - 0.10)):
                db_gallery_entries = self._get_db_gallery_from_ann()
                max_linear_search = 100
                search_entries = db_gallery_entries[:max_linear_search] if len(db_gallery_entries) > max_linear_search else db_gallery_entries
                for entry in search_entries:
                    entry_id = entry.get("id")
                    if entry_id in gallery_ids or entry_id in used_ids:
                        continue
                    centroid = entry.get("centroid")
                    if centroid is None:
                        continue
                    sim = float(np.dot(embedding, centroid))
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry
        else:
            # Linear search without ANN
            for entry in self.db_gallery:
                entry_id = entry["id"]
                if entry_id in gallery_ids or entry_id in used_ids:
                    continue
                centroid = entry.get("centroid")
                if centroid is None:
                    continue
                sim = float(np.dot(embedding, centroid))
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

        if best_entry is None:
            return None, best_sim
        
        db_recall_threshold = max(self.similarity_threshold - 0.05, 0.48)
        if best_sim < db_recall_threshold:
            return None, best_sim

        # Return a copy
        revived_entry = {
            "id": best_entry["id"],
            "centroid": best_entry["centroid"].copy(),
            "prototypes": [best_entry["centroid"].copy()],
            "count": best_entry.get("count", 0),
            "last_seen": best_entry.get("last_seen", 0),
            "last_seen_ts": best_entry.get("last_seen_ts", int(time.time())),
            "missed_frames": best_entry.get("missed_frames", 0),
            "hits": best_entry.get("hits", 0),
            "quality": best_entry.get("quality", 0.0),
            "created_at": best_entry.get("created_at", int(time.time())),
            "bbox": None,
        }
        self.next_id = max(self.next_id, revived_entry["id"] + 1)
        return revived_entry, best_sim


    # ========== Main Assignment Logic ==========

    def assign(self, detections_info, frame_idx, total_detections):
        assignment = [None] * total_detections
        if detections_info is None:
            detections_info = []

        # Process and normalize embeddings
        processed = []
        for det_idx, det in enumerate(detections_info):
            det_index = det.get("det_index", det_idx)
            emb = self._normalize(det.get("embedding"))
            if emb is None:
                continue
            processed.append({
                "det_index": det_index,
                "embedding": emb,
                "bbox": det.get("bbox"),
                "quality": float(det.get("quality", 1.0) or 1.0),
            })

        # Increment missed frames for all gallery entries
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

            # Stage 1: Search in ANN candidates (fast search from gallery + db_gallery)
            candidate_entries = self._get_ann_candidates(det["embedding"], frame_idx)
            best_entry, best_sim = self._search_in_entries(det["embedding"], candidate_entries, used_ids)
            
            # If no match in candidates, search in full gallery (fallback for better accuracy)
            if best_entry is None or best_sim < self.similarity_threshold:
                gallery_entry, gallery_sim = self._search_in_entries(det["embedding"], self.gallery, used_ids)
                if gallery_sim > best_sim:
                    best_entry = gallery_entry
                    best_sim = gallery_sim

            # Match found?
            if best_entry is not None and best_sim >= self.similarity_threshold:
                # Update entry in gallery
                if best_entry["id"] in {e["id"] for e in self.gallery}:
                    self._update_entry(best_entry, det["embedding"], frame_idx, det.get("bbox"), det.get("quality", 1.0))
                    self._persist_entry(best_entry)
                else:
                    # Add revived entry from DB to gallery
                    best_entry["last_seen"] = frame_idx
                    best_entry["last_seen_ts"] = int(time.time())
                    best_entry["bbox"] = det.get("bbox")
                    best_entry["quality"] = max(best_entry.get("quality", 0.0), det.get("quality", 1.0))
                    best_entry["missed_frames"] = 0
                    best_entry["hits"] = best_entry.get("hits", 0) + 1
                    self.gallery.append(best_entry)
                    self._persist_entry(best_entry)
                    self._ann_dirty = True
                
                assignment[det_index] = best_entry["id"]
                used_ids.add(best_entry["id"])
                continue

            # Stage 2: No match found, search more broadly in db_gallery (for revive)
            # Only search in db_gallery (gallery already searched in Stage 1)
            revived_entry, revived_sim = self._revive_from_db(det["embedding"], used_ids, frame_idx)
            if revived_entry is not None:
                revived_entry["last_seen"] = frame_idx
                revived_entry["last_seen_ts"] = int(time.time())
                revived_entry["bbox"] = det.get("bbox")
                revived_entry["quality"] = max(revived_entry.get("quality", 0.0), det.get("quality", 1.0))
                revived_entry["missed_frames"] = 0
                revived_entry["hits"] = revived_entry.get("hits", 0) + 1
                self.gallery.append(revived_entry)
                self._persist_entry(revived_entry)
                self._ann_dirty = True
                assignment[det_index] = revived_entry["id"]
                used_ids.add(revived_entry["id"])
            else:
                # Create new ID
                new_entry = self._create_entry(det["embedding"], frame_idx, det.get("bbox"), det.get("quality", 1.0))
                assignment[det_index] = new_entry["id"]
                used_ids.add(new_entry["id"])

        self._prune()
        return assignment
