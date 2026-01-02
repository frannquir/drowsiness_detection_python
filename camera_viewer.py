import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from collections import deque
import threading

try:
    import winsound
except ImportError:
    winsound = None


class CameraViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Viewer - Face Detection")

        self.camera = None
        self.current_camera_index = 0
        self.is_running = True

        # Initialize MediaPipe Face Landmarker
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # Blink detection variables (improved precision)
        self.blink_threshold = 0.21
        self.BLINK_MIN_FRAMES = 2
        self.blink_counter = 0
        self.blink_frames = 0
        self.is_blinking = False

        # Yawn detection variables (simplified - MAR only)
        self.YAWN_START_THRESHOLD = 0.35  # Lowered from 0.45
        self.YAWN_END_THRESHOLD = 0.25  # Lowered from 0.35
        self.YAWN_MIN_FRAMES = 10  # Increased to require mouth open longer
        self.yawn_counter = 0
        self.yawn_frames = 0
        self.is_yawning = False

        # Frontal face gate
        self.eye_span_baseline = None
        self.FRONTAL_THRESHOLD = 0.75

        # Warning for prolonged eye closure
        self.eyes_closed_start = None
        self.warning_active = False
        self.warning_event_fired = False  # Track if event registered for current closure

        # Nod detection constants (downward-only approach)
        self.NOSE_IDX = 1
        self.DIP_THRESHOLD = 0.035
        self.VEL_THRESHOLD = 0.08
        self.MIN_DIP_TIME = 0.10
        self.COOLDOWN = 1.0
        self.FACE_LOSS_GRACE = 0.6
        self.LOSS_AFTER_DIP_WINDOW = 0.7
        self.PEAK_CONFIRM = 0.045

        # Nod detection state (downward dip tracking)
        self.nod_counter = 0
        self.nose_y_s = None  # smoothed nose Y
        self.nose_baseline = None
        self.prev_nose_y_s = None
        self.prev_nose_t = None
        self.last_face_time = None
        self.nod_cooldown_until = 0
        self.pending_dip = False
        self.dip_time = None
        self.dip_peak_delta = 0.0
        self.nod_flash_until = 0

        # Drowsiness detection constants
        self.WARN_WINDOW_SHORT = 60.0
        self.WARN_WINDOW_LONG = 180.0
        self.NOD_WINDOW = 60.0
        self.LONG_EYES_CLOSED_SEC = 4.0  # Reduced from 6s to 4s
        self.DROWSY_DISPLAY_SEC = 4.0

        # Drowsiness detection state (WARNING-based only)
        self.warn_times = deque()
        self.drowsy_until = 0.0
        self.long_close_triggered = False

        # Alarm escalation system (wake-up alarm style)
        self.STABLE_AWAKE_SEC = 2.0
        self.ALARM_START_INTERVAL = 8.0  # Start at 8s instead of 10s
        self.ALARM_MIN_INTERVAL = 1.5  # Minimum 1.5s instead of 2.5s
        self.ALARM_ACCEL_EVERY = 3.0  # Escalate every 3s instead of 8s
        self.ALARM_LEVEL_MAX = 8  # More levels for gradual escalation
        self.alarm_active = False
        self.alarm_level = 0
        self.next_alarm_time = 0.0
        self.awake_since = None
        self.last_risky_time = 0.0

        # Nod alarm system (triggers on single nod)
        self.nod_alarm_active = False
        self.nod_alarm_level = 0
        self.nod_next_beep_time = 0.0
        self.nod_last_escalation = 0.0
        self.nod_awake_since = None

        # Detect available cameras
        self.available_cameras = self.detect_cameras()

        # Create UI
        self.create_ui()

        # Start camera
        if self.available_cameras:
            self.start_camera(self.available_cameras[0])

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start update loop
        self.update_frame()

    def detect_cameras(self):
        """Detect all available cameras"""
        available = []
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def create_ui(self):
        """Create minimal UI with camera selector"""
        # Control panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Camera selector
        tk.Label(control_frame, text="Camera:").pack(side=tk.LEFT, padx=5)

        self.camera_var = tk.StringVar()
        camera_options = [f"Camera {i}" for i in self.available_cameras]

        if camera_options:
            self.camera_var.set(camera_options[0])

        camera_dropdown = ttk.Combobox(
            control_frame,
            textvariable=self.camera_var,
            values=camera_options,
            state="readonly",
            width=15
        )
        camera_dropdown.pack(side=tk.LEFT, padx=5)
        camera_dropdown.bind("<<ComboboxSelected>>", self.on_camera_change)

        # Video display
        self.video_label = tk.Label(self.root)
        self.video_label.pack(expand=True, fill=tk.BOTH)

    def start_camera(self, camera_index):
        """Start camera capture"""
        if self.camera is not None:
            self.camera.release()

        self.current_camera_index = camera_index
        self.camera = cv2.VideoCapture(camera_index)

        if self.camera.isOpened():
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def on_camera_change(self, event):
        """Handle camera selection change"""
        selected = self.camera_var.get()
        camera_index = int(selected.split()[-1])
        self.start_camera(camera_index)

    def calculate_yawn_symmetry(self, landmarks):
        """Calculate mouth opening symmetry (left vs right)"""
        # Left lip opening: 78 (upper) to 81 (lower)
        left_open = np.linalg.norm(
            np.array([landmarks[78].x, landmarks[78].y]) -
            np.array([landmarks[81].x, landmarks[81].y])
        )
        # Right lip opening: 308 (upper) to 311 (lower)
        right_open = np.linalg.norm(
            np.array([landmarks[308].x, landmarks[308].y]) -
            np.array([landmarks[311].x, landmarks[311].y])
        )
        # Symmetry ratio
        if max(left_open, right_open) == 0:
            return 0.0
        return min(left_open, right_open) / max(left_open, right_open)

    def is_face_frontal(self, landmarks):
        """Check if face is frontal enough (using inter-ocular distance)"""
        # Inter-ocular distance: 33 (left outer) to 263 (right outer)
        eye_span = np.linalg.norm(
            np.array([landmarks[33].x, landmarks[33].y]) -
            np.array([landmarks[263].x, landmarks[263].y])
        )

        # Update baseline with slow EMA
        if self.eye_span_baseline is None:
            self.eye_span_baseline = eye_span
            return True
        else:
            # Update baseline slowly when face is stable
            beta = 0.01
            self.eye_span_baseline = beta * eye_span + (1 - beta) * self.eye_span_baseline

        # Check if current span is close to baseline
        return eye_span >= (self.FRONTAL_THRESHOLD * self.eye_span_baseline)

    def play_beep_async(self, freq=1200, dur_ms=500):
        """Play beep sound asynchronously (non-blocking)"""
        if winsound is None:
            return
        # Run beep in background thread so Tkinter loop doesn't stall
        threading.Thread(target=winsound.Beep, args=(freq, dur_ms), daemon=True).start()

    def prune_deque(self, dq, now, window):
        """Remove timestamps older than window from deque"""
        while dq and (now - dq[0]) > window:
            dq.popleft()

    def calculate_ear(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio for blink detection"""
        # Get eye landmarks
        points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])

        # Compute vertical distances
        vertical1 = np.linalg.norm(points[1] - points[5])
        vertical2 = np.linalg.norm(points[2] - points[4])

        # Compute horizontal distance
        horizontal = np.linalg.norm(points[0] - points[3])

        # Calculate EAR
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear

    def calculate_mar(self, landmarks, mouth_indices):
        """Calculate Mouth Aspect Ratio for yawn detection"""
        # Get mouth landmarks
        points = np.array([[landmarks[i].x, landmarks[i].y] for i in mouth_indices])

        # Compute vertical distances
        vertical1 = np.linalg.norm(points[1] - points[7])
        vertical2 = np.linalg.norm(points[2] - points[6])
        vertical3 = np.linalg.norm(points[3] - points[5])

        # Compute horizontal distance
        horizontal = np.linalg.norm(points[0] - points[4])

        # Calculate MAR
        mar = (vertical1 + vertical2 + vertical3) / (3.0 * horizontal)
        return mar

    def update_frame(self):
        """Update video frame"""
        if self.is_running and self.camera is not None and self.camera.isOpened():
            ret, frame = self.camera.read()

            if ret:
                # Convert BGR to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create MediaPipe Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # Process face landmarks
                detection_result = self.landmarker.detect(mp_image)

                # Define landmark indices for eyes and mouth
                LEFT_EYE = [33, 160, 158, 133, 153, 144]
                RIGHT_EYE = [362, 385, 387, 263, 373, 380]
                # Mouth: [left_corner, top1, top2, top3, right_corner, bottom3, bottom2, bottom1]
                # Fixed: proper top-to-bottom pairs for accurate MAR calculation
                MOUTH = [61, 13, 82, 312, 291, 317, 87, 14]

                # Track blink and yawn status
                blink_detected = False
                yawn_detected = False

                # Current time for nod detection
                now = time.monotonic()

                if detection_result.face_landmarks:
                    for face_landmarks in detection_result.face_landmarks:
                        # Update last face time
                        self.last_face_time = now

                        # Draw ONLY keypoints for eyes, mouth, and nose
                        h, w, _ = frame.shape
                        keypoints = set(LEFT_EYE + RIGHT_EYE + MOUTH + [self.NOSE_IDX])
                        for idx in keypoints:
                            landmark = face_landmarks[idx]
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                        # Track nose Y for nod detection (downward-only)
                        nose_y = face_landmarks[self.NOSE_IDX].y

                        # Smooth nose position with EMA
                        alpha = 0.25
                        if self.nose_y_s is None:
                            self.nose_y_s = nose_y
                        else:
                            self.nose_y_s = alpha * nose_y + (1 - alpha) * self.nose_y_s

                        # Update baseline when not in pending dip
                        if not self.pending_dip:
                            beta = 0.02
                            if self.nose_baseline is None:
                                self.nose_baseline = self.nose_y_s
                            else:
                                self.nose_baseline = beta * self.nose_y_s + (1 - beta) * self.nose_baseline

                        # Calculate downward velocity
                        vel = 0.0
                        if self.prev_nose_y_s is not None and self.prev_nose_t is not None:
                            dt = max(now - self.prev_nose_t, 1e-3)
                            vel = (self.nose_y_s - self.prev_nose_y_s) / dt

                        # Store for next frame
                        self.prev_nose_y_s = self.nose_y_s
                        self.prev_nose_t = now

                        # Calculate EAR for both eyes
                        left_ear = self.calculate_ear(face_landmarks, LEFT_EYE)
                        right_ear = self.calculate_ear(face_landmarks, RIGHT_EYE)
                        avg_ear = (left_ear + right_ear) / 2.0

                        # Calculate MAR for mouth
                        mar = self.calculate_mar(face_landmarks, MOUTH)

                        # Improved blink detection: require BOTH eyes closed
                        both_eyes_closed = (left_ear < self.blink_threshold) and (right_ear < self.blink_threshold)
                        blink_detected = False

                        if both_eyes_closed:
                            self.blink_frames += 1
                            if self.blink_frames >= self.BLINK_MIN_FRAMES and not self.is_blinking:
                                # Count blink once
                                self.blink_counter += 1
                                self.is_blinking = True
                                blink_detected = True
                        else:
                            self.blink_frames = 0
                            self.is_blinking = False

                        # Track prolonged eye closure for warning
                        eyes_closed = (avg_ear < self.blink_threshold)
                        if eyes_closed:
                            if self.eyes_closed_start is None:
                                self.eyes_closed_start = time.monotonic()

                            eyes_closed_duration = time.monotonic() - self.eyes_closed_start

                            # 2-second WARNING (register event once per closure)
                            if eyes_closed_duration >= 2.0:
                                self.warning_active = True
                                if not self.warning_event_fired:
                                    # Register warning event only once per closure
                                    self.warn_times.append(now)
                                    self.warning_event_fired = True
                            else:
                                self.warning_active = False

                            # 6-second LONG CLOSE -> triggers alarm activation below
                            if eyes_closed_duration >= self.LONG_EYES_CLOSED_SEC:
                                if not self.long_close_triggered:
                                    self.long_close_triggered = True
                        else:
                            self.eyes_closed_start = None
                            self.warning_active = False
                            self.warning_event_fired = False
                            self.long_close_triggered = False

                        # Yawn detection: simplified MAR-only approach
                        yawn_detected = False

                        # Check yawn conditions with hysteresis
                        if self.is_yawning:
                            # Already yawning - use lower threshold to end
                            yawn_active = mar > self.YAWN_END_THRESHOLD
                        else:
                            # Not yawning - check if MAR exceeds start threshold
                            yawn_active = mar > self.YAWN_START_THRESHOLD

                        if yawn_active:
                            self.yawn_frames += 1
                            if self.yawn_frames >= self.YAWN_MIN_FRAMES and not self.is_yawning:
                                # Count yawn once - passed all gates (UI only, not for DROWSY)
                                self.yawn_counter += 1
                                self.is_yawning = True
                                yawn_detected = True
                        else:
                            self.yawn_frames = 0
                            self.is_yawning = False

                        # Nod detection: downward dip logic
                        if self.nose_baseline is not None and now >= self.nod_cooldown_until:
                            delta = self.nose_y_s - self.nose_baseline

                            # Check if downward dip is happening
                            if delta > self.DIP_THRESHOLD and vel > self.VEL_THRESHOLD:
                                if not self.pending_dip:
                                    self.pending_dip = True
                                    self.dip_time = now
                                    self.dip_peak_delta = delta
                                else:
                                    self.dip_peak_delta = max(self.dip_peak_delta, delta)

                            # Check if we should count the nod (downward confirmed)
                            if self.pending_dip:
                                dip_duration = now - self.dip_time
                                if dip_duration >= self.MIN_DIP_TIME and self.dip_peak_delta >= self.PEAK_CONFIRM:
                                    # Count nod and trigger wake-up beep
                                    self.nod_counter += 1
                                    self.nod_cooldown_until = now + self.COOLDOWN
                                    self.nod_flash_until = now + 0.4
                                    self.pending_dip = False
                                    # Play immediate wake-up beep on nod
                                    self.play_beep_async(1000, 300)
                                    # Activate nod alarm
                                    self.nod_alarm_active = True
                                    self.nod_alarm_level = 0
                                    self.nod_next_beep_time = now + 5.0  # Start checking in 5s
                                    self.nod_last_escalation = now
                                    self.nod_awake_since = None
                                elif dip_duration > 1.5:
                                    # Reset if too long
                                    self.pending_dip = False

                        # Define risky/awake state for alarm system
                        face_detected = True  # We're inside face_landmarks block
                        risky = (not face_detected) or both_eyes_closed
                        awake = face_detected and (not both_eyes_closed)

                        # Alarm activation trigger (WARNING-based only)
                        if not self.alarm_active:
                            # Prune old warning events
                            self.prune_deque(self.warn_times, now, self.WARN_WINDOW_LONG)

                            # Count warnings in time windows
                            warn_60 = sum(1 for t in self.warn_times if (now - t) <= self.WARN_WINDOW_SHORT)
                            warn_180 = len(self.warn_times)

                            # Check trigger conditions
                            should_activate = False
                            if eyes_closed and eyes_closed_duration >= self.LONG_EYES_CLOSED_SEC:
                                should_activate = True
                            elif warn_60 >= 2 or warn_180 >= 3:
                                should_activate = True

                            if should_activate:
                                # Activate alarm
                                self.alarm_active = True
                                self.alarm_level = 0
                                self.next_alarm_time = now  # Beep immediately
                                self.awake_since = None
                                self.last_risky_time = now
                                self.drowsy_until = now + self.DROWSY_DISPLAY_SEC

                        # Alarm escalation loop (while alarm is active)
                        if self.alarm_active:
                            if awake:
                                # User is awake - beep faster to help wake up fully
                                if self.awake_since is None:
                                    self.awake_since = now

                                # Check if fully awake (disarm after 2 seconds)
                                if (now - self.awake_since) >= self.STABLE_AWAKE_SEC:
                                    # Disarm completely
                                    self.alarm_active = False
                                    self.alarm_level = 0
                                    self.next_alarm_time = 0
                                    self.awake_since = None
                                    self.last_risky_time = 0.0
                                    self.warn_times.clear()
                                    self.drowsy_until = 0  # Stop DROWSY display
                                else:
                                    # Eyes open but not yet stable - beep FAST to ensure wakefulness
                                    if now >= self.next_alarm_time:
                                        # Triple beep for wake-up urgency
                                        self.play_beep_async(1400, 150)
                                        threading.Timer(0.2, lambda: self.play_beep_async(1400, 150)).start()
                                        threading.Timer(0.4, lambda: self.play_beep_async(1400, 150)).start()
                                        # Very short interval (1 second) when eyes open
                                        self.next_alarm_time = now + 1.0
                                        self.drowsy_until = now + self.DROWSY_DISPLAY_SEC
                            else:
                                # Still risky - reset awake timer, escalate alarm
                                self.awake_since = None

                                # Escalate alarm level over time
                                if self.last_risky_time == 0.0:
                                    self.last_risky_time = now
                                elif (now - self.last_risky_time) >= self.ALARM_ACCEL_EVERY:
                                    self.alarm_level = min(self.ALARM_LEVEL_MAX, self.alarm_level + 1)
                                    self.last_risky_time = now

                                # Beep scheduler
                                if now >= self.next_alarm_time:
                                    # Play beep (escalated pattern for higher levels)
                                    if self.alarm_level >= 3:
                                        # Double beep for urgency
                                        self.play_beep_async(1200, 200)
                                        threading.Timer(0.25, lambda: self.play_beep_async(1200, 200)).start()
                                    else:
                                        # Single beep
                                        self.play_beep_async(1200, 500)

                                    # Calculate next alarm interval (faster as level increases)
                                    # More aggressive decrease: -1.0s per level instead of -1.5s per level
                                    interval = max(self.ALARM_MIN_INTERVAL,
                                                  self.ALARM_START_INTERVAL - self.alarm_level * 1.0)
                                    self.next_alarm_time = now + interval
                                    self.drowsy_until = now + self.DROWSY_DISPLAY_SEC  # Keep showing DROWSY

                        # Nod alarm escalation (triggers on single nod, escalates if face lost or eyes closed)
                        if self.nod_alarm_active:
                            # Check if risky: face lost or eyes closed
                            nod_risky = (not face_detected) or both_eyes_closed
                            nod_awake = face_detected and (not both_eyes_closed)

                            if nod_awake:
                                # Face detected and eyes open - start disarm timer
                                if self.nod_awake_since is None:
                                    self.nod_awake_since = now
                                elif (now - self.nod_awake_since) >= self.STABLE_AWAKE_SEC:
                                    # Disarm nod alarm after 2s of being awake
                                    self.nod_alarm_active = False
                                    self.nod_alarm_level = 0
                                    self.nod_awake_since = None
                            else:
                                # Still risky (face lost or eyes closed) - escalate
                                self.nod_awake_since = None

                                # Escalate level every 3 seconds
                                if (now - self.nod_last_escalation) >= self.ALARM_ACCEL_EVERY:
                                    self.nod_alarm_level = min(self.ALARM_LEVEL_MAX, self.nod_alarm_level + 1)
                                    self.nod_last_escalation = now

                                # Beep scheduler
                                if now >= self.nod_next_beep_time:
                                    # Play escalating beeps
                                    if self.nod_alarm_level >= 3:
                                        # Double beep
                                        self.play_beep_async(1000, 200)
                                        threading.Timer(0.25, lambda: self.play_beep_async(1000, 200)).start()
                                    else:
                                        # Single beep
                                        self.play_beep_async(1000, 400)

                                    # Calculate interval (gets faster with level)
                                    interval = max(self.ALARM_MIN_INTERVAL,
                                                  6.0 - self.nod_alarm_level * 0.8)
                                    self.nod_next_beep_time = now + interval

                        # Display metrics with yawn debug info
                        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        yawn_color = (0, 255, 255) if yawn_active else (0, 255, 0)
                        cv2.putText(frame, f"MAR: {mar:.2f} [{self.yawn_frames}/10]", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, yawn_color, 2)

                        # Display counters
                        cv2.putText(frame, f"Blinks: {self.blink_counter}", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(frame, f"Yawns: {self.yawn_counter}", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(frame, f"Nods: {self.nod_counter}", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                        # Display status
                        if blink_detected:
                            cv2.putText(frame, "BLINK!", (10, 190),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                        if yawn_detected:
                            cv2.putText(frame, "YAWN!", (10, 230),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

                        # Display nod flash
                        if now < self.nod_flash_until:
                            cv2.putText(frame, "NOD!", (10, 270),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

                        # Display WARNING when eyes closed >= 3 seconds
                        if self.warning_active:
                            cv2.putText(frame, "WARNING!", (10, 310),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

                        # Display DROWSY alert if active
                        if now < self.drowsy_until:
                            cv2.putText(frame, "DROWSY!", (10, 360),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)

                else:
                    # Handle face tracking loss gracefully
                    face_detected = False
                    risky = True  # Face loss is risky
                    awake = False

                    # Handle alarm when face is lost
                    if self.alarm_active:
                        # Face lost while alarm active - keep escalating
                        self.awake_since = None

                        # Escalate alarm level
                        if self.last_risky_time == 0.0:
                            self.last_risky_time = now
                        elif (now - self.last_risky_time) >= self.ALARM_ACCEL_EVERY:
                            self.alarm_level = min(self.ALARM_LEVEL_MAX, self.alarm_level + 1)
                            self.last_risky_time = now

                        # Beep scheduler
                        if now >= self.next_alarm_time:
                            if self.alarm_level >= 3:
                                self.play_beep_async(1200, 200)
                                threading.Timer(0.25, lambda: self.play_beep_async(1200, 200)).start()
                            else:
                                self.play_beep_async(1200, 500)

                            interval = max(self.ALARM_MIN_INTERVAL,
                                          self.ALARM_START_INTERVAL - self.alarm_level * 1.0)
                            self.next_alarm_time = now + interval
                            self.drowsy_until = now + self.DROWSY_DISPLAY_SEC

                    # Nod alarm when face is lost (escalates since face loss = risky)
                    if self.nod_alarm_active:
                        # Face lost means risky - escalate
                        self.nod_awake_since = None

                        # Escalate level
                        if (now - self.nod_last_escalation) >= self.ALARM_ACCEL_EVERY:
                            self.nod_alarm_level = min(self.ALARM_LEVEL_MAX, self.nod_alarm_level + 1)
                            self.nod_last_escalation = now

                        # Beep scheduler
                        if now >= self.nod_next_beep_time:
                            if self.nod_alarm_level >= 3:
                                self.play_beep_async(1000, 200)
                                threading.Timer(0.25, lambda: self.play_beep_async(1000, 200)).start()
                            else:
                                self.play_beep_async(1000, 400)

                            interval = max(self.ALARM_MIN_INTERVAL,
                                          6.0 - self.nod_alarm_level * 0.8)
                            self.nod_next_beep_time = now + interval

                    # Nod detection during face loss
                    if self.last_face_time is not None:
                        time_since_face = now - self.last_face_time

                        if time_since_face <= self.FACE_LOSS_GRACE:
                            # Within grace period - check if we had a pending dip
                            if self.pending_dip and self.dip_time is not None:
                                time_since_dip = now - self.dip_time
                                if time_since_dip <= self.LOSS_AFTER_DIP_WINDOW and now >= self.nod_cooldown_until:
                                    # Count nod and trigger wake-up beep
                                    self.nod_counter += 1
                                    self.nod_cooldown_until = now + self.COOLDOWN
                                    self.nod_flash_until = now + 0.4
                                    self.pending_dip = False
                                    # Play immediate wake-up beep on nod
                                    self.play_beep_async(1000, 300)
                                    # Activate nod alarm
                                    self.nod_alarm_active = True
                                    self.nod_alarm_level = 0
                                    self.nod_next_beep_time = now + 5.0  # Start checking in 5s
                                    self.nod_last_escalation = now
                                    self.nod_awake_since = None
                        else:
                            # Beyond grace period - reset pending dip
                            self.pending_dip = False

                # Convert frame to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image
                img = Image.fromarray(frame_rgb)

                # Convert to PhotoImage
                imgtk = ImageTk.PhotoImage(image=img)

                # Update label
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        # Schedule next update
        if self.is_running:
            self.root.after(10, self.update_frame)

    def on_closing(self):
        """Clean up on window close"""
        self.is_running = False
        if self.camera is not None:
            self.camera.release()
        self.landmarker.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = CameraViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
