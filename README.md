# Recycling Detector

This project is a **Recycling Detector** that uses YOLOv8 to detect and classify different types of materials used in recyclable and non-recyclable items. The model is trained to identify **seven** different material types: cardboard, glass, materials, metal, paper, plastic, and trash.

## Project Structure

The dataset is divided into training, validation, and testing sets:

- **Training Images:** `../train/images`
- **Validation Images:** `../valid/images`
- **Testing Images:** `../test/images`

### Classes
The model is trained on the following seven classes:

1. `cardboard`
2. `glass`
3. `materials`
4. `metal`
5. `paper`
6. `plastic`
7. `trash`

## Dataset Information

The dataset for training and evaluation is sourced from Roboflow's **Recyclables and Garbage Detection** project.

- **Roboflow Workspace:** [recycling-detector](https://universe.roboflow.com/recycling-detector/recyclables-and-garbage-detection)
- **Project:** Recyclables and Garbage Detection
- **Version:** 4
- **License:** MIT
- **Dataset URL:** [Link to Dataset](https://universe.roboflow.com/recycling-detector/recyclables-and-garbage-detection/dataset/4)

## Getting Started

### Prerequisites

To run the recycling detector, you will need the following:

- **Python 3.8+**
- **YOLOv8** (from `ultralytics`)
- **OpenCV** for video capturing and image processing

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/recycling-detector.git
   ```

2. Install the required dependencies:

   ```bash
   pip install ultralytics opencv-python
   ```

3. Download the dataset from Roboflow and place the images in their respective folders (`../train/images`, `../valid/images`, and `../test/images`).

### Model Training

To train the YOLOv8 model, use the following command:

```bash
from ultralytics import YOLO

# Load your custom YOLOv8 model configuration
model = YOLO("yolov8n.yaml")

# Train the model
model.train(data="path/to/your/data.yaml", epochs=100)
```

Make sure to update the `data.yaml` file with the correct paths to the dataset.

### Running the Model

Once the model is trained, you can use it for real-time material detection through your webcam or on a set of images.

Here’s a sample code to run the detector on a webcam:

```python
import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("./runs/detect/exp/weights/best.pt")  # Replace with your trained model path

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(source=frame)

    # Annotate the frame with bounding boxes
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('Recycling Detector', annotated_frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
```

### Testing the Model

To test the model on the test dataset, use the following command:

```bash
# Load the trained model and run it on the test set
results = model.val(data="path/to/your/data.yaml")
```

## Results

After training, the model will output metrics such as **mAP (mean Average Precision)** to evaluate its performance on the test data. The predictions will include bounding boxes around detected materials along with the confidence score for each classification.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset is sourced from Roboflow's **Recyclables and Garbage Detection** project.
- YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics).

---

Feel free to modify and adapt this project to suit your needs!