# pi-face

Real-time face recognition application with MQTT notifications.

## Usage

Install dependencies from `requirements.txt` and ensure your camera is available to OpenCV. You can start the application with either of the following commands from the repository root:

```bash
python -m src.main [--once] [--camera-index N] [--display]
```

or

```bash
python src/main.py [--once] [--camera-index N] [--display]
```

Both entry points initialize logging, connect to the database, and begin processing camera frames. Use `--once` to process a single frame for testing, `--camera-index` to choose a different camera, and `--display` to open a debug window with bounding boxes.
