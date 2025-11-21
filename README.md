# Hailo Face ID Detection

Real-time face recognition system for Raspberry Pi 5 with Hailo AI accelerator. Provides persistent person identification across video frames using deep learning embeddings and an optimized tracking system.

**Note**: The code requires Hailo SDK installed on the system to run. Import errors for `hailo_apps` during static analysis are expected and do not indicate code issues.

## 🚀 Quick Start

```bash
# Set environment variables
export HAILO_DEVICE_IDS="0001:01:00.0"
export HAILORT_VDEVICE_GROUP_ID="1"

# Run the application
cd ~/hailo-faceid-detection/hailo-rpi5-examples
./install.sh
source .env
export DISPLAY=0
export PYTHONPATH=/home/pi/hailo-faceid-detection/hailo-rpi5-examples:/home/pi/hailo-apps-infra:$PYTHONPATH
python3 basic_pipelines/detection_faceid.py
```

## Key Features

* **Real-time Face Recognition**: High FPS processing using Hailo AI accelerator
* **Persistent ID Tracking**: Maintains person identities across sessions using SQLite
* **ANN-based Fast Search**: Optional Approximate Nearest Neighbor indexing
* **Asynchronous Database Operations**: Non-blocking DB writes for smooth performance
* **Multi-stage Search Strategy**: Optimized 2-stage search for accurate ID assignment

## Requirements

* Raspberry Pi 5
* Hailo AI accelerator (Hailo-8)
* Python 3.8+
* Hailo Runtime libraries

## Installation

```bash
./install.sh
pip install numpy opencv-python hailo-platform annoy
```

## Configuration

Set environment variables for customization:

```bash
export REID_USE_ANN="1"              # Enable ANN (0 or 1)
export REID_ANN_CANDIDATES="20"      # ANN candidate count
export REID_ANN_TREES="10"           # ANN tree count
export REID_DB_RECALL_LIMIT="200"    # Max DB entries to load
export REID_ANN_REBUILD_INTERVAL="30" # ANN rebuild interval (frames)
```

## Documentation

For complete documentation, see [DOCUMENTATION.md](DOCUMENTATION.md).

The documentation includes:
* Detailed architecture overview
* API reference
* Configuration guide
* Performance optimization tips
* Troubleshooting guide
* Usage examples

## Architecture

```
GStreamer → Person Detection → FaceIDPipeline → Tracker → SQLite DB
                                    ↓
                              ArcFace Model
                              (Embeddings)
```

## Project Structure

```
hailo-rpi5-examples/
├── basic_pipelines/
│   ├── utils.py                    # Tracker class
│   ├── detection_faceid.py         # Main application
│   └── detection_faceid_pipeline.py # FaceIDPipeline class
├── data/
│   └── faceid.sqlite              # SQLite database
└── DOCUMENTATION.md               # Complete documentation
```

## Usage Example

```python
from basic_pipelines.detection_faceid import user_app_callback_class, app_callback
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

user_data = user_app_callback_class()
app = GStreamerDetectionApp(app_callback, user_data)
app.run()
```

## Troubleshooting

Common issues and solutions are documented in [DOCUMENTATION.md#troubleshooting](DOCUMENTATION.md#troubleshooting).

## License

Part of the Hailo Raspberry Pi 5 examples repository.

---

**For detailed documentation, see [DOCUMENTATION.md](DOCUMENTATION.md)**
