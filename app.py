from fer import FER
import cv2
import time
from flask import Flask, request, jsonify

app = Flask(__name__)
emotion_detector = FER(mtcnn=True)

def predict_image(img):

    # Convert Gradio image format (BGR) to RGB
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    analysis = emotion_detector.detect_emotions(img)
    start_time = time.time() 

    res = []
    for data in analysis:
      emotions = data['emotions']
      max_emotion = max(emotions, key=emotions.get)
      max_value = emotions[max_emotion]
      res.append({max_emotion: max_value})
  
    response = {
        'analysis': analysis,
        'best_predict': list(res),
    }
    return response

@app.route('/', methods=['GET'])
def index():
    res = {"app": "Text sentiment analysis for social media"}
    return jsonify(res)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img_path = 'temp.jpg'  # Temporarily save the image to a file
    file.save(img_path)
    
    result = predict_image(img_path)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run()
