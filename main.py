
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import math

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/traffic.mp4")

if not cap.isOpened():
    print("Error: Cannot open video!")
    exit()


trackers = {}

next_id = 1

MAX_DISTANCE = 120
MAX_MISSED_FRAMES = 15

VEHICLE_CLASSES = [2, 3, 5, 7]

LINE_POSITION = 0.60

up_count = 0
down_count = 0

class VehicleTracker:

    def __init__(self, vehicle_id, center, class_id):

        self.id = vehicle_id
        self.class_id = class_id

        self.kalman = cv2.KalmanFilter(4, 2)

        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kalman.processNoiseCov = (
            np.eye(4, dtype=np.float32) * 0.03
        )

        self.kalman.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * 0.1
        )

        self.kalman.errorCovPost = (
            np.eye(4, dtype=np.float32)
        )

        self.kalman.statePost = np.array([
            [np.float32(center[0])],
            [np.float32(center[1])],
            [0],
            [0]
        ], dtype=np.float32)

        self.position = center
        self.box = None

        self.missed = 0

        self.last_observed_y = center[1]
        self.previous_y = center[1]

        self.counted = False
        self.direction = None

    def predict(self):

        prediction = self.kalman.predict()

        self.position = (
            int(prediction[0][0]),
            int(prediction[1][0])
        )

        self.missed += 1

        return self.position

    def update(self, center, box):

        measurement = np.array([
            [np.float32(center[0])],
            [np.float32(center[1])]
        ])

        self.kalman.correct(measurement)

        self.previous_y = self.last_observed_y

        self.last_observed_y = center[1]

        self.position = center
        self.box = box

        self.missed = 0


print("Vehicle Counting System Started!")


while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    LINE_Y = int(height * LINE_POSITION)

    results = model(
        frame,
        conf=0.35,
        classes=VEHICLE_CLASSES,
        verbose=False
    )

    detections = []

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        class_id = int(box.cls[0])

        detections.append({
            "center": (cx, cy),
            "box": (x1, y1, x2, y2),
            "class_id": class_id
        })

    tracker_ids = list(trackers.keys())

    predictions = []

    for tid in tracker_ids:

        predictions.append(
            trackers[tid].predict()
        )


    matched_detections = set()

    if len(detections) > 0 and len(trackers) > 0:

        cost_matrix = np.full(
            (len(detections), len(trackers)),
            100000.0
        )

        for i, detection in enumerate(detections):

            cx, cy = detection["center"]

            for j, tid in enumerate(tracker_ids):

                tracker = trackers[tid]

                # Match same vehicle class
                if detection["class_id"] != tracker.class_id:
                    continue

                px, py = predictions[j]

                distance = math.sqrt(
                    (cx - px) ** 2 +
                    (cy - py) ** 2
                )

                cost_matrix[i, j] = distance

        rows, cols = linear_sum_assignment(cost_matrix)

        for row, col in zip(rows, cols):

            if cost_matrix[row, col] < MAX_DISTANCE:

                tid = tracker_ids[col]

                detection = detections[row]

                trackers[tid].update(
                    detection["center"],
                    detection["box"]
                )

                matched_detections.add(row)


    for i, detection in enumerate(detections):

        if i not in matched_detections:

            trackers[next_id] = VehicleTracker(
                next_id,
                detection["center"],
                detection["class_id"]
            )

            trackers[next_id].box = detection["box"]

            next_id += 1

    for tid in list(trackers.keys()):

        if trackers[tid].missed > MAX_MISSED_FRAMES:

            del trackers[tid]

    cv2.line(
        frame,
        (0, LINE_Y),
        (width, LINE_Y),
        (0, 255, 255),
        3
    )

    cv2.putText(
        frame,
        "COUNTING LINE",
        (20, LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    for tid, tracker in trackers.items():

        if tracker.missed > 0:
            continue

        if tracker.box is None:
            continue

        cx, cy = tracker.position

        if not tracker.counted:

            if (
                tracker.previous_y < LINE_Y
                and tracker.last_observed_y >= LINE_Y
            ):

                down_count += 1

                tracker.counted = True
                tracker.direction = "Downward"

                print(f"Vehicle {tid}: DOWNWARD")

            elif (
                tracker.previous_y > LINE_Y
                and tracker.last_observed_y <= LINE_Y
            ):

                up_count += 1

                tracker.counted = True
                tracker.direction = "Upward"

                print(f"Vehicle {tid}: UPWARD")


        x1, y1, x2, y2 = tracker.box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (cx, cy),
            4,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"ID: {tid}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    total_count = up_count + down_count

    cv2.rectangle(
        frame,
        (10, 10),
        (310, 125),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Upward: {up_count}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Downward: {down_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Total: {total_count}",
        (20, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Automated Vehicle Counting System",
        frame
    )

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFINAL VEHICLE COUNT")

print("Upward:", up_count)
print("Downward:", down_count)
print("Total:", up_count + down_count)

print("Vehicle Counting System Stopped")
