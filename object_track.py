import threading
import time
import sqlite3
import os
from tkinter import Tk, Button, Label, filedialog, StringVar, simpledialog, messagebox, Scale, IntVar, Frame

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("OpenCV (cv2) not found. Install with: pip install opencv-python") from e


class DetectionDB:
    def __init__(self, db_path="detections.sqlite"):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    roi_name TEXT,
                    timestamp REAL NOT NULL,
                    x INTEGER,
                    y INTEGER,
                    w INTEGER,
                    h INTEGER,
                    fps REAL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rois (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    source TEXT,
                    x INTEGER,
                    y INTEGER,
                    w INTEGER,
                    h INTEGER,
                    created_at REAL
                );
                """
            )
            # Migrate detections table to add roi_name if missing
            cur.execute("PRAGMA table_info(detections)")
            cols = [r[1] for r in cur.fetchall()]
            if "roi_name" not in cols:
                cur.execute("ALTER TABLE detections ADD COLUMN roi_name TEXT")
            conn.commit()
        finally:
            conn.close()

    def log(self, source, roi_name, timestamp, x, y, w, h, fps):
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO detections (source, roi_name, timestamp, x, y, w, h, fps) VALUES (?,?,?,?,?,?,?,?)",
                (source, roi_name, timestamp, x, y, w, h, fps),
            )
            conn.commit()
        finally:
            conn.close()

    def save_roi(self, name, source, x, y, w, h):
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO rois (name, source, x, y, w, h, created_at) VALUES (?,?,?,?,?,?,?)",
                (name, source, x, y, w, h, time.time()),
            )
            conn.commit()
        finally:
            conn.close()


class TrackerApp:
    def __init__(self):
        self.root = Tk()
        self.root.title("Object Tracking - Webcam / Video / Image")
        self.status_var = StringVar()
        self.status_var.set("Idle")

        # GUI size variables
        self.gui_scale_var = IntVar(value=100)
        self.opencv_scale_var = IntVar(value=100)
        self.image_scale_var = IntVar(value=100)

        # Create main frame
        main_frame = Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=8, pady=8)

        # Size control frame
        size_frame = Frame(main_frame)
        size_frame.pack(fill="x", pady=(0, 8))
        
        Label(size_frame, text="GUI Size Controls:", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # GUI scale control
        gui_scale_frame = Frame(size_frame)
        gui_scale_frame.pack(fill="x", pady=2)
        Label(gui_scale_frame, text="Main GUI Scale:").pack(side="left")
        Scale(gui_scale_frame, from_=50, to=200, orient="horizontal", 
              variable=self.gui_scale_var, length=150, command=self.update_gui_size).pack(side="left", padx=(5, 0))
        Label(gui_scale_frame, text="%").pack(side="left")
        
        # OpenCV window scale control
        opencv_scale_frame = Frame(size_frame)
        opencv_scale_frame.pack(fill="x", pady=2)
        Label(opencv_scale_frame, text="Tracking Window Scale:").pack(side="left")
        Scale(opencv_scale_frame, from_=25, to=200, orient="horizontal", 
              variable=self.opencv_scale_var, length=150).pack(side="left", padx=(5, 0))
        Label(opencv_scale_frame, text="%").pack(side="left")
        
        # Image detection scale control
        image_scale_frame = Frame(size_frame)
        image_scale_frame.pack(fill="x", pady=2)
        Label(image_scale_frame, text="Image Detection Scale:").pack(side="left")
        Scale(image_scale_frame, from_=25, to=200, orient="horizontal", 
              variable=self.image_scale_var, length=150).pack(side="left", padx=(5, 0))
        Label(image_scale_frame, text="%").pack(side="left")

        self.btn_webcam = Button(main_frame, text="Track from Webcam", command=self.start_webcam)
        self.btn_video = Button(main_frame, text="Track from Video", command=self.start_video)
        self.btn_image = Button(main_frame, text="Track in Image", command=self.start_image)
        self.btn_stop = Button(main_frame, text="Stop", command=self.stop_tracking)
        self.lbl_status = Label(main_frame, textvariable=self.status_var)

        # Instructions and threshold slider
        self.instructions = Label(
            main_frame,
            text=(
                "How to use:\n"
                "1) Click Webcam/Video/Image.\n"
                "2) Draw ROI, press Enter/Space.\n"
                "3) Enter ROI name.\n"
                "4) Choose where to track (same/webcam/video/image).\n"
                "Tip: Adjust threshold if tracking flickers."
            ),
            justify="left"
        )
        self.threshold_var = IntVar(value=40)
        self.slider = Scale(
            main_frame,
            from_=0,
            to=255,
            orient="horizontal",
            label="Detection Threshold",
            variable=self.threshold_var,
            length=300
        )

        self.instructions.pack(padx=8, pady=(8, 2))
        self.slider.pack(padx=8, pady=(0, 8))
        self.btn_webcam.pack(padx=8, pady=6)
        self.btn_video.pack(padx=8, pady=6)
        self.btn_image.pack(padx=8, pady=6)
        self.btn_stop.pack(padx=8, pady=6)
        self.lbl_status.pack(padx=8, pady=6)

        self.db = DetectionDB()

        self.tracking_thread = None
        self.stop_event = threading.Event()
        self.current_source = None
        self.current_roi_name = None
        self.last_roi_meta = None  # {name, hist, base_size:(w,h), base_frame:(W,H)}
        self.detect_threshold = 40  # backprojection confidence threshold (0-255)

    def update_gui_size(self, value):
        """Update the main GUI size based on scale"""
        scale = int(value) / 100.0
        # Update font sizes and window geometry
        self.root.geometry(f"{int(400 * scale)}x{int(500 * scale)}")

    def get_opencv_scale(self):
        """Get the OpenCV window scale factor"""
        return self.opencv_scale_var.get() / 100.0

    def get_image_scale(self):
        """Get the image detection scale factor"""
        return self.image_scale_var.get() / 100.0

    def compute_initial_window(self, frame, roi_hist, roi_size, base_frame_size=None):
        base_w, base_h = roi_size
        fw, fh = frame.shape[1], frame.shape[0]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 60, 32), (180, 255, 255))
        back_proj = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
        back_proj = cv2.bitwise_and(back_proj, back_proj, mask=mask)
        back_proj = cv2.GaussianBlur(back_proj, (9, 9), 0)
        # Determine base scale
        scale = 1.0
        if base_frame_size is not None:
            bw, bh = base_frame_size
            if bw > 0 and bh > 0:
                scale = min(fw / bw, fh / bh)
        # Try a few scales around the estimate
        best = None
        for s in (0.85 * scale, 1.0 * scale, 1.2 * scale):
            w = max(10, int(base_w * s))
            h = max(10, int(base_h * s))
            _, maxVal, _, maxLoc = cv2.minMaxLoc(back_proj)
            cx, cy = maxLoc
            x = max(0, min(fw - w, cx - w // 2))
            y = max(0, min(fh - h, cy - h // 2))
            score = float(maxVal) / (w * h)
            if best is None or score > best[0]:
                best = (score, int(x), int(y), int(w), int(h))
        _, x, y, w, h = best
        return (x, y, w, h)

    def set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def start_webcam(self):
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.set_status("Already tracking. Stop first.")
            return
        self.current_source = "webcam"
        self.stop_event.clear()
        # Run preparation on main thread (thread-safe Tk)
        self.track_from_webcam()

    def start_video(self):
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.set_status("Already tracking. Stop first.")
            return
        path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv")])
        if not path:
            return
        self.current_source = path
        self.stop_event.clear()
        # Run preparation on main thread (thread-safe Tk)
        self.track_from_video(path)

    def start_image(self):
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.set_status("Already tracking. Stop first.")
            return
        path = filedialog.askopenfilename(title="Select Image", filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not path:
            return
        self.current_source = path
        self.stop_event.clear()
        # Run image flow in main thread to keep Tk dialogs thread-safe
        self.track_in_image(path)

    def stop_tracking(self):
        self.stop_event.set()
        self.set_status("Stopping...")

    def select_initial_roi(self, frame):
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        roi = cv2.selectROI("Select ROI", frame, False, False)
        cv2.destroyWindow("Select ROI")
        if roi and all(roi):
            x, y, w, h = roi
            return int(x), int(y), int(w), int(h)
        return None

    def prompt_roi_name(self):
        name = simpledialog.askstring("ROI Name", "Enter a name for the selected ROI:", parent=self.root)
        if name is None or not str(name).strip():
            messagebox.showwarning("ROI Name", "ROI name is required to proceed.")
            return None
        return str(name).strip()

    def build_roi_hist(self, frame, roi_rect):
        x, y, w, h = roi_rect
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return None
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Mask to ignore low saturation/value regions to reduce noise
        mask = cv2.inRange(hsv_roi, (0, 60, 32), (180, 255, 255))
        # Use H and S channels histogram for better discrimination
        roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask, [180, 256], [0, 180, 0, 256])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        return roi_hist

    def run_camshift_loop(self, cap, source_label, roi_name, roi_hist, track_window):
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1)

        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        # Set initial window size based on scale
        scale = self.get_opencv_scale()
        cv2.resizeWindow("Tracking", int(640 * scale), int(480 * scale))
        start_time = time.time()
        frames = 0
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            elapsed = max(1e-6, time.time() - start_time)
            fps_val = frames / elapsed

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, 60, 32), (180, 255, 255))
            back_proj = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
            # Apply mask and smooth back projection to stabilize tracking
            back_proj = cv2.bitwise_and(back_proj, back_proj, mask=mask)
            back_proj = cv2.GaussianBlur(back_proj, (5, 5), 0)
            # Confidence check
            _, maxVal, _, _ = cv2.minMaxLoc(back_proj)
            # Read live threshold from GUI
            self.detect_threshold = int(self.threshold_var.get()) if hasattr(self, 'threshold_var') else self.detect_threshold
            if maxVal < self.detect_threshold:
                cv2.putText(frame, "No detection", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # CamShift adapts window size/angle
                ret, track_window = cv2.CamShift(back_proj, track_window, term_crit)
                pts = cv2.boxPoints(ret)
                pts = np.int32(pts)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            cv2.putText(frame, f"{roi_name} | {source_label} FPS: {fps_val:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Approximate bounding rect for logging
            x, y, w, h = track_window
            self.db.log(source_label, roi_name, time.time(), int(x), int(y), int(w), int(h), fps_val)

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        self.set_status("Idle")

    def open_webcam(self):
        # Try multiple indices and Windows backends for robustness
        backends = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)
        if hasattr(cv2, "CAP_ANY"):
            backends.append(cv2.CAP_ANY)
        indices = [0, 1, 2, 3]
        for idx in indices:
            for backend in backends:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    return cap
                cap.release()
        # Final fallback
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def track_from_webcam(self):
        self.set_status("Starting webcam...")
        cap = self.open_webcam()
        if not cap or not cap.isOpened():
            self.set_status("Cannot open webcam. Ensure it's connected and not in use.")
            messagebox.showerror("Webcam", "Cannot open webcam. Try reconnecting or closing other apps using the camera.")
            return
        # Prepare ROI and histogram on main thread for thread-safe dialogs
        ok, frame = cap.read()
        if not ok:
            self.set_status("Failed to read from webcam.")
            cap.release()
            return
        roi_rect = self.select_initial_roi(frame)
        if roi_rect is None:
            self.set_status("ROI selection canceled.")
            cap.release()
            return
        roi_name = self.prompt_roi_name()
        if roi_name is None:
            self.set_status("ROI naming canceled.")
            cap.release()
            return
        roi_hist = self.build_roi_hist(frame, roi_rect)
        if roi_hist is None:
            self.set_status("Invalid ROI.")
            cap.release()
            return
        track_window = tuple(roi_rect)
        self.db.save_roi(roi_name, "webcam", track_window[0], track_window[1], track_window[2], track_window[3])
        # Store last ROI for cross-source scaling
        self.last_roi_meta = {
            "name": roi_name,
            "hist": roi_hist,
            "base_size": (track_window[2], track_window[3]),
            "base_frame": (frame.shape[1], frame.shape[0]),
        }
        target_kind, target_path = self.choose_target_after_roi("webcam")
        if target_kind is None:
            cap.release()
            self.set_status("Tracking canceled.")
            return
        if target_kind == "same":
            # Start background loop using prepared data
            self.stop_event.clear()
            self.tracking_thread = threading.Thread(
                target=self.run_camshift_loop,
                args=(cap, "webcam", roi_name, roi_hist, track_window),
                daemon=True,
            )
            self.tracking_thread.start()
        elif target_kind == "webcam":
            # Open a new webcam stream and initialize using the prepared ROI histogram
            cap.release()
            new_cap = self.open_webcam()
            if not new_cap or not new_cap.isOpened():
                self.set_status("Cannot open webcam for target.")
                return
            ok2, frame2 = new_cap.read()
            if not ok2:
                self.set_status("Failed to read from webcam target.")
                new_cap.release()
                return
            init_window = self.compute_initial_window(frame2, roi_hist, (track_window[2], track_window[3]), self.last_roi_meta["base_frame"])
            self.stop_event.clear()
            self.tracking_thread = threading.Thread(
                target=self.run_camshift_loop,
                args=(new_cap, "webcam", roi_name, roi_hist, init_window),
                daemon=True,
            )
            self.tracking_thread.start()
        elif target_kind == "video":
            # Open selected video and initialize using the prepared ROI histogram
            cap.release()
            vcap = cv2.VideoCapture(target_path)
            if not vcap.isOpened():
                self.set_status("Cannot open selected video.")
                return
            ok2, frame2 = vcap.read()
            if not ok2:
                self.set_status("Failed to read from selected video.")
                vcap.release()
                return
            init_window = self.compute_initial_window(frame2, roi_hist, (track_window[2], track_window[3]), self.last_roi_meta["base_frame"])
            self.stop_event.clear()
            self.tracking_thread = threading.Thread(
                target=self.run_camshift_loop,
                args=(vcap, os.path.basename(target_path), roi_name, roi_hist, init_window),
                daemon=True,
            )
            self.tracking_thread.start()
        elif target_kind == "image":
            # Detect once on image using the prepared ROI histogram
            cap.release()
            img = cv2.imread(target_path)
            if img is None:
                self.set_status("Cannot open selected image.")
                return
            init_window = self.compute_initial_window(img, roi_hist, (track_window[2], track_window[3]), self.last_roi_meta["base_frame"])
            x, y, w0, h0 = init_window
            cv2.rectangle(img, (x, y), (x + w0, y + h0), (0, 255, 0), 2)
            cv2.putText(img, roi_name, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self.db.save_roi(roi_name, os.path.basename(target_path), x, y, w0, h0)
            self.db.log(os.path.basename(target_path), roi_name, time.time(), x, y, w0, h0, 0.0)
            cv2.namedWindow("Image Detection", cv2.WINDOW_NORMAL)
            # Set window size based on scale
            scale = self.get_image_scale()
            cv2.resizeWindow("Image Detection", int(img.shape[1] * scale), int(img.shape[0] * scale))
            cv2.imshow("Image Detection", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.set_status("Idle")

    def track_from_video(self, path):
        self.set_status(f"Opening video: {os.path.basename(path)}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.set_status("Cannot open video")
            return
        # Prepare ROI and histogram on main thread for thread-safe dialogs
        ok, frame = cap.read()
        if not ok:
            self.set_status("Failed to read from video.")
            cap.release()
            return
        roi_rect = self.select_initial_roi(frame)
        if roi_rect is None:
            self.set_status("ROI selection canceled.")
            cap.release()
            return
        roi_name = self.prompt_roi_name()
        if roi_name is None:
            self.set_status("ROI naming canceled.")
            cap.release()
            return
        roi_hist = self.build_roi_hist(frame, roi_rect)
        if roi_hist is None:
            self.set_status("Invalid ROI.")
            cap.release()
            return
        track_window = tuple(roi_rect)
        src_label = os.path.basename(path)
        self.db.save_roi(roi_name, src_label, track_window[0], track_window[1], track_window[2], track_window[3])
        # Store last ROI for cross-source scaling
        self.last_roi_meta = {
            "name": roi_name,
            "hist": roi_hist,
            "base_size": (track_window[2], track_window[3]),
            "base_frame": (frame.shape[1], frame.shape[0]),
        }
        target_kind, target_path = self.choose_target_after_roi(src_label)
        if target_kind is None:
            cap.release()
            self.set_status("Tracking canceled.")
            return
        if target_kind == "same":
            # If same on image source, allow picking another image file
            if src_label.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                cap.release()
                img_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp")])
                if not img_path:
                    self.set_status("Tracking canceled.")
                    return
                img = cv2.imread(img_path)
                if img is None:
                    self.set_status("Cannot open selected image.")
                    return
                init_window = self.compute_initial_window(img, roi_hist, (track_window[2], track_window[3]), self.last_roi_meta["base_frame"])
                x, y, w0, h0 = init_window
                cv2.rectangle(img, (x, y), (x + w0, y + h0), (0, 255, 0), 2)
                cv2.putText(img, roi_name, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                self.db.save_roi(roi_name, os.path.basename(img_path), x, y, w0, h0)
                self.db.log(os.path.basename(img_path), roi_name, time.time(), x, y, w0, h0, 0.0)
                cv2.namedWindow("Image Detection", cv2.WINDOW_NORMAL)
                # Set window size based on scale
                scale = self.get_image_scale()
                cv2.resizeWindow("Image Detection", int(img.shape[1] * scale), int(img.shape[0] * scale))
                cv2.imshow("Image Detection", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                self.set_status("Idle")
                return
            self.stop_event.clear()
            self.tracking_thread = threading.Thread(
                target=self.run_camshift_loop,
                args=(cap, src_label, roi_name, roi_hist, track_window),
                daemon=True,
            )
            self.tracking_thread.start()
        elif target_kind == "webcam":
            cap.release()
            new_cap = self.open_webcam()
            if not new_cap or not new_cap.isOpened():
                self.set_status("Cannot open webcam for target.")
                return
            ok2, frame2 = new_cap.read()
            if not ok2:
                self.set_status("Failed to read from webcam target.")
                new_cap.release()
                return
            init_window = self.compute_initial_window(frame2, roi_hist, (track_window[2], track_window[3]))
            self.stop_event.clear()
            self.tracking_thread = threading.Thread(
                target=self.run_camshift_loop,
                args=(new_cap, "webcam", roi_name, roi_hist, init_window),
                daemon=True,
            )
            self.tracking_thread.start()
        elif target_kind == "video":
            cap.release()
            vcap = cv2.VideoCapture(target_path)
            if not vcap.isOpened():
                self.set_status("Cannot open selected video.")
                return
            ok2, frame2 = vcap.read()
            if not ok2:
                self.set_status("Failed to read from selected video.")
                vcap.release()
                return
            init_window = self.compute_initial_window(frame2, roi_hist, (track_window[2], track_window[3]), self.last_roi_meta["base_frame"])
            self.stop_event.clear()
            self.tracking_thread = threading.Thread(
                target=self.run_camshift_loop,
                args=(vcap, os.path.basename(target_path), roi_name, roi_hist, init_window),
                daemon=True,
            )
            self.tracking_thread.start()
        elif target_kind == "image":
            cap.release()
            img = cv2.imread(target_path)
            if img is None:
                self.set_status("Cannot open selected image.")
                return
            init_window = self.compute_initial_window(img, roi_hist, (track_window[2], track_window[3]), self.last_roi_meta["base_frame"])
            x, y, w0, h0 = init_window
            cv2.rectangle(img, (x, y), (x + w0, y + h0), (0, 255, 0), 2)
            cv2.putText(img, roi_name, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self.db.save_roi(roi_name, os.path.basename(target_path), x, y, w0, h0)
            self.db.log(os.path.basename(target_path), roi_name, time.time(), x, y, w0, h0, 0.0)
            cv2.namedWindow("Image Detection", cv2.WINDOW_NORMAL)
            # Set window size based on scale
            scale = self.get_image_scale()
            cv2.resizeWindow("Image Detection", int(img.shape[1] * scale), int(img.shape[0] * scale))
            cv2.imshow("Image Detection", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.set_status("Idle")

    def choose_target_after_roi(self, current_source_label):
        use_same = messagebox.askyesno("Tracking Target", f"Track on the same source ({current_source_label})?")
        if use_same:
            return ("same", None)
        choice = simpledialog.askstring("Tracking Target", "Enter target: webcam, video, or image", parent=self.root)
        if not choice:
            return (None, None)
        choice = choice.strip().lower()
        if choice == "webcam":
            return ("webcam", None)
        if choice == "video":
            path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv")])
            if not path:
                return (None, None)
            return ("video", path)
        if choice == "image":
            path = filedialog.askopenfilename(title="Select Image", filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp")])
            if not path:
                return (None, None)
            return ("image", path)
        return (None, None)

    def track_in_image(self, path):
        self.set_status(f"Opening image: {os.path.basename(path)}")
        img = cv2.imread(path)
        if img is None:
            self.set_status("Cannot open image")
            return
        roi_rect = self.select_initial_roi(img)
        if roi_rect is None:
            self.set_status("ROI selection canceled.")
            return
        roi_name = self.prompt_roi_name()
        if roi_name is None:
            self.set_status("ROI naming canceled.")
            return
        x, y, w, h = roi_rect
        roi_hist = self.build_roi_hist(img, roi_rect)
        if roi_hist is None:
            self.set_status("Invalid ROI.")
            return
        self.db.save_roi(roi_name, os.path.basename(path), x, y, w, h)
        # Persist ROI meta for cross-source scaling
        self.last_roi_meta = {
            "name": roi_name,
            "hist": roi_hist,
            "base_size": (w, h),
            "base_frame": (img.shape[1], img.shape[0]),
        }
        # Offer target selection after ROI from image
        target_kind, target_path = self.choose_target_after_roi(os.path.basename(path))
        if target_kind is None:
            # Default: show detection on this image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, roi_name, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self.db.log(os.path.basename(path), roi_name, time.time(), x, y, w, h, 0.0)
            cv2.namedWindow("Image Detection", cv2.WINDOW_NORMAL)
            # Set window size based on scale
            scale = self.get_image_scale()
            cv2.resizeWindow("Image Detection", int(img.shape[1] * scale), int(img.shape[0] * scale))
            cv2.imshow("Image Detection", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.set_status("Idle")
            return
        if target_kind == "same":
            # Detect on this same image using histogram correlation to find best window
            init_window = self.compute_initial_window(img, roi_hist, (w, h), self.last_roi_meta["base_frame"])
            ix, iy, iw, ih = init_window
            cv2.rectangle(img, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 2)
            cv2.putText(img, roi_name, (ix, max(0, iy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self.db.log(os.path.basename(path), roi_name, time.time(), ix, iy, iw, ih, 0.0)
            cv2.namedWindow("Image Detection", cv2.WINDOW_NORMAL)
            # Set window size based on scale
            scale = self.get_image_scale()
            cv2.resizeWindow("Image Detection", int(img.shape[1] * scale), int(img.shape[0] * scale))
            cv2.imshow("Image Detection", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.set_status("Idle")
        elif target_kind == "webcam":
            new_cap = self.open_webcam()
            if not new_cap or not new_cap.isOpened():
                self.set_status("Cannot open webcam for target.")
                return
            ok2, frame2 = new_cap.read()
            if not ok2:
                self.set_status("Failed to read from webcam target.")
                new_cap.release()
                return
            init_window = self.compute_initial_window(frame2, roi_hist, (w, h), self.last_roi_meta["base_frame"])
            self.stop_event.clear()
            self.tracking_thread = threading.Thread(
                target=self.run_camshift_loop,
                args=(new_cap, "webcam", roi_name, roi_hist, init_window),
                daemon=True,
            )
            self.tracking_thread.start()
        elif target_kind == "video":
            vcap = cv2.VideoCapture(target_path)
            if not vcap.isOpened():
                self.set_status("Cannot open selected video.")
                return
            ok2, frame2 = vcap.read()
            if not ok2:
                self.set_status("Failed to read from selected video.")
                vcap.release()
                return
            init_window = self.compute_initial_window(frame2, roi_hist, (w, h), self.last_roi_meta["base_frame"])
            self.stop_event.clear()
            self.tracking_thread = threading.Thread(
                target=self.run_camshift_loop,
                args=(vcap, os.path.basename(target_path), roi_name, roi_hist, init_window),
                daemon=True,
            )
            self.tracking_thread.start()
        elif target_kind == "image":
            img2 = cv2.imread(target_path)
            if img2 is None:
                self.set_status("Cannot open selected image.")
                return
            init_window = self.compute_initial_window(img2, roi_hist, (w, h), self.last_roi_meta["base_frame"])
            ix, iy, iw, ih = init_window
            cv2.rectangle(img2, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 2)
            cv2.putText(img2, roi_name, (ix, max(0, iy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self.db.save_roi(roi_name, os.path.basename(target_path), ix, iy, iw, ih)
            self.db.log(os.path.basename(target_path), roi_name, time.time(), ix, iy, iw, ih, 0.0)
            cv2.namedWindow("Image Detection", cv2.WINDOW_NORMAL)
            # Set window size based on scale
            scale = self.get_image_scale()
            cv2.resizeWindow("Image Detection", int(img2.shape[1] * scale), int(img2.shape[0] * scale))
            cv2.imshow("Image Detection", img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.set_status("Idle")

    def run(self):
        self.root.mainloop()


def main():
    app = TrackerApp()
    app.run()


if __name__ == "__main__":
    main()


