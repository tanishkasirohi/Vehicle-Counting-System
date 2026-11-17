# Automated Vehicle Detection, Tracking and Counting System

An automated vehicle counting system developed using Python, OpenCV and YOLOv8. The application detects vehicles in traffic surveillance videos, assigns unique IDs, tracks their movement using Kalman Filters and counts vehicles crossing a predefined virtual line in opposite directions.

## Project Overview

Traffic monitoring plays an important role in understanding road usage and managing congestion. This project uses computer vision to automate vehicle detection, tracking and counting from surveillance footage.

The system processes video frames, identifies vehicles, tracks their positions and calculates the number of vehicles moving in different directions.

## Features

- Vehicle detection using YOLOv8.
- Unique identification numbers for detected vehicles.
- Object tracking using Kalman Filters.
- Vehicle matching using the Hungarian Algorithm.
- Virtual line-based vehicle counting.
- Detection of upward and downward vehicle movement.
- Real-time display of vehicle counts.

## Technologies Used

- Python
- OpenCV
- YOLOv8
- NumPy
- SciPy
- Kalman Filter
- Hungarian Algorithm

## How It Works

1. A traffic surveillance video is loaded into the application.
2. YOLO detects vehicles in consecutive frames.
3. Each detected vehicle is assigned a unique ID.
4. A Kalman Filter predicts vehicle positions.
5. The Hungarian Algorithm matches detections with existing tracks.
6. A virtual counting line identifies vehicles crossing the line.
7. Vehicles are counted according to their direction of movement.
8. The total count is displayed in real time.

## Project Structure

```text
Automated-Vehicle-Detection-and-Counting/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/tanishkasirohi/Vehicle-Counting-System
```

Navigate to the project folder:

```bash
cd Vehicle-Counting-System
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Input Video

Place a traffic surveillance video inside a folder named `videos` and rename it to:

```text
traffic.mp4
```

The YOLOv8 model is downloaded automatically during its first execution.

## Run the Application

```bash
python main.py
```

Press `q` to close the video window.

## Output

The application displays:

- Bounding boxes around detected vehicles.
- Unique vehicle IDs.
- Vehicle centre positions.
- A virtual counting line.
- Upward vehicle count.
- Downward vehicle count.
- Total number of vehicles counted.

## Future Improvements

- Improve tracking accuracy in crowded traffic conditions.
- Reduce ID switching during vehicle occlusion.
- Support multiple counting lines.
- Generate traffic analysis reports.
- Add vehicle classification and traffic statistics.

## Author

**Tanishka Sirohi**
GitHub: [tanishkasirohi](https://github.com/tanishkasirohi)
