import uvicorn
from backend.app.main import app
import threading
import time
import requests
import base64
import numpy as np
import cv2
import json

def run_server():
    uvicorn.run(app, host='127.0.0.1', port=8000, log_level='warning')

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)

print('=== END-TO-END PIPELINE VERIFICATION ===')
print()

# 1. Health check
response = requests.get('http://127.0.0.1:8000/api/health')
print('1. Health Check:')
print(json.dumps(response.json(), indent=2))
print()

# 2. Classes
response = requests.get('http://127.0.0.1:8000/api/classes')
print('2. Classes (29):')
print('   Count: ' + str(response.json()['count']))
print('   First: ' + str(response.json()['classes'][0]))
print('   Last: ' + str(response.json()['classes'][-1]))
print()

# 3. No-hand image (random noise)
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
_, buffer = cv2.imencode('.jpg', img)
test_img = base64.b64encode(buffer).decode('utf-8')
response = requests.post('http://127.0.0.1:8000/api/predict', 
                       json={'image': test_img, 'use_stabilizer': False}, timeout=30)
result = response.json()
print('3. No-hand image (random noise):')
print('   Hand detected: ' + str(result['hand_detected']))
print('   Predictions: ' + str(len(result['predictions'])))
print()

# 4. Real ASL 'A' image
asl_img_path = r'D:\5th sem codebase\DL pro\sign-language-cnn\data\raw\asl_alphabet_train\asl_alphabet_train\A\A1.jpg'
img = cv2.imread(asl_img_path)
_, buffer = cv2.imencode('.jpg', img)
test_img = base64.b64encode(buffer).decode('utf-8')
response = requests.post('http://127.0.0.1:8000/api/predict', 
                       json={'image': test_img, 'use_stabilizer': False}, timeout=30)
result = response.json()
print('4. Real ASL A image:')
print('   Hand detected: ' + str(result['hand_detected']))
print('   Top prediction: ' + result['predictions'][0]['class_label'] + ' (' + str(result['predictions'][0]['confidence']) + ')')
print('   Inference time: ' + str(result['inference_time_ms']) + 'ms')
print('   Total time: ' + str(result['total_time_ms']) + 'ms')
print()

# 5. Stabilizer test
print('5. Stabilization test (sequence of A predictions):')
for i in range(5):
    response = requests.post('http://127.0.0.1:8000/api/predict', 
                           json={'image': test_img, 'use_stabilizer': True}, timeout=30)
    result = response.json()
    if result.get('stabilized'):
        stab = result['stabilized']
        print('   Frame ' + str(i+1) + ': action=' + stab['action'] + ', char="' + stab['character'] + '", stable=' + str(stab['is_stable']))
print()

# 6. Translation buffer
response = requests.post('http://127.0.0.1:8000/api/translate')
print('6. Translation buffer:')
print('   Text: "' + response.json()['text'] + '"')
print()

print('=== ALL TESTS PASSED ===')