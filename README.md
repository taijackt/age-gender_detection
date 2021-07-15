# Age / Gender detection

### Steps:
- First use yolo to detect face position then crop the face and classify the gender and age group.

### Dependencies:
- torch >= 1.4.0
- onnxruntime

### How to run:
- Open the python file `demo.py`, go to `line 216`, change the parameter `source` to either rtsp protocol or -1 for webcam
- Activate youe environment and run `python demo.py` on your terminal.
