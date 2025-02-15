import tensorflow.lite as tflite
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Updated labels based on model training
labels = ["Optimal_Moisture", "Moderate_Moisture", "Low_Moisture", "Not_A_Cucumber"]

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        image = preprocess_image(image)

        # Perform inference
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        # Get predicted class and confidence
        predicted_class = labels[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        # If model predicts "Not_A_Cucumber", return an appropriate response
        if predicted_class == "Not_A_Cucumber":
            return jsonify({
                "is_cucumber": False,
                "message": "No cucumber detected in the image."
            }), 200  

        # Define lifespan based on moisture level
        lifespan_mapping = {
            "Optimal_Moisture": (7, 9),
            "Moderate_Moisture": (5, 7),
            "Low_Moisture": (3, 5)
        }
        lifespan_range = lifespan_mapping.get(predicted_class, (0, 0))
        
        return jsonify({
            "is_cucumber": True,
            "moisture_level": predicted_class,
            "lifespan": f"{lifespan_range[0]}-{lifespan_range[1]} days",
            "confidence": f"{confidence:.2f}%"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
