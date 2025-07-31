from flask import Flask, Response, render_template_string # type: ignore
from flask_httpauth import HTTPBasicAuth # type: ignore
import threading
import io
import serial # For serial communication with Arduino
from datetime import datetime
from RPLCD.i2c import CharLCD
from gpiozero import Buzzer
import time
# import Adafruit Blinka
import board
import digitalio
# import Adafruit IO REST client.
from Adafruit_IO import Client, Feed, RequestError
from picamera2 import Picamera2
import numpy as np
import tensorflow.lite as tflite
from PIL import Image
import cv2
import socket

# Attempt to initialize LCD
def initialize_lcd():
    try:
        from RPLCD.i2c import CharLCD
        return CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=20, rows=4, dotsize=8)
    except Exception as e:
        print(f"LCD initialization failed: {e}")
        return None

lcd = initialize_lcd()

def initialize_arduino():
    try:
        return serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    except Exception as e:
        print(f"Arduino initialization failed: {e}")
        return None

arduino = initialize_arduino()

# Initialize Picamera2
# Function to check and initialize Picamera2
def initialize_camera():
    try:
        camera = Picamera2()
        camera.configure(camera.create_still_configuration())
        camera.start()
        return camera
    except Exception as e:
        print(f"Camera initialization failed: {e}")
        return None
    
picam2 = initialize_camera()

# Load TFLite model
interpreter = tflite.Interpreter(model_path="/home/thesis/Downloads/Pi Files/model_unquant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

ADAFRUIT_IO_KEY = "YOUR_ADAFRUIT_IO_KEY"
ADAFRUIT_IO_USERNAME = "YOUR_ADAFRUIT_IO_USERNAME"

aio = Client(ADAFRUIT_IO_USERNAME, ADAFRUIT_IO_KEY)

pH_Feed = aio.feeds('aquathesis.ph')
oxy_Feed = aio.feeds('aquathesis.oxygen')
temp_Feed = aio.feeds('aquathesis.temperature')
light_Feed = aio.feeds('aquathesis.light')
botWater_Feed = aio.feeds('aquathesis.water-level-lower')
topWater_Feed = aio.feeds('aquathesis.water-level-upper')
camera_Feed = aio.feeds('aquathesis.camera')

# button set up
buzzer = Buzzer(18)
time.sleep(2)  # Allow time for connection to establish

# Flask app for online streaming
app = Flask(__name__)
auth = HTTPBasicAuth()

users = {
    "aquaponics": "thesis"
}

@auth.get_password
def get_pw(username):
    return users.get(username)

# Stream-only page (requires login)
@app.route('/stream')
@auth.login_required
def stream():
    return '<h2>Live Stream</h2><img src="/video">'

@app.route('/video')
@auth.login_required
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# HTML template page (no login)
HTML_TEMPLATE = """
<!doctype html>
<html>
    <head>
        <title>Live Stream</title>
    </head>
    <body>
        <h1>Live Stream</h1>
        <img src="{{ url_for('video_feed') }}" width="100%">
    </body>
</html>
"""

@app.route('/')
def index():
    """Main page of the streaming server."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Route to serve the video feed."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Generator function to yield video frames."""
    while True:
        frame = picam2.capture_array()  # Capture a frame as a NumPy array
        # Convert the frame to JPEG
        img = Image.fromarray(frame)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_bytes.read() + b'\r\n')

# Start Flask app
def start_streaming():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def parse_data(data):
    """
    Parse the sensor data received from the Arduino.
    Args:
        data (str): Comma-separated sensor data string.
    Returns:
        dict: Parsed sensor values in a dictionary.
    """
    sensor_values = {}
    try:
        pairs = data.split(",")  # Split the string into key-value pairs
        for pair in pairs:
            key, value = pair.split(":")  # Split each pair into key and value
            sensor_values[key.strip()] = float(value.strip())  # Convert values to float
    except ValueError:
        print("Error parsing data:", data)
    return sensor_values

# Process image
def preprocess_image(image_array, input_shape):
    image = Image.fromarray(image_array).resize((input_shape[1], input_shape[2]))
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

# Perform inference
def predict(image_array):
    input_shape = input_details[0]['shape']
    input_data = preprocess_image(image_array, input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    label, confidence = get_prediction_label(output_data, labels)
    return label, confidence

def load_labels(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels("/home/thesis/Downloads/Pi Files/labels.txt")

def get_prediction_label(output_data, labels):
    max_index = np.argmax(output_data)  # Get the index of the highest probability
    return labels[max_index], output_data[max_index]  # Return the label and confidence

def get_ip_address():
    """
    Get the current IP address of the Raspberry Pi.
    Returns:
        str: The IP address of the Raspberry Pi, or a message if unable to get the IP.
    """
    try:
        # Create a socket and connect to a remote address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Use a public DNS server to determine the local IP
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except Exception as e:
        return f"Unable to get IP address: {e}"
    
def check_internet(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return 1  # Internet connected
    except socket.error:
        return 0  # No internet

try:
    if __name__ == "__main__":
        ip = get_ip_address()
        print(f"Raspberry Pi IP Address: {ip}")
    
    flask_thread = threading.Thread(target=start_streaming)
    flask_thread.daemon = True
    flask_thread.start()
    while True:
        lcd = lcd or initialize_lcd()  # Continuously check for LCD availability
        arduino = arduino or initialize_arduino()  # Continuously check for Arduino availability
        picam2 = picam2 or initialize_camera()  # Continuously check for Camera availability
        # Send the current time to the Arduino
        if arduino:
            current_time = datetime.now().strftime("Time:%H\n")
            arduino.write(current_time.encode('utf-8'))
            print(f"Sent to Arduino: {current_time.strip()}")
        
        if picam2:
            frame = picam2.capture_array()
            label, confidence = predict(frame)
            plantHealth = f"Prediction: {label} ({confidence:.2f})"
            
        internet_status = check_internet()
        internet_msg = f"Internet:{internet_status}\n"
        arduino.write(internet_msg.encode('utf-8'))
        print(f"Sent to Arduino: {internet_msg.strip()}")

        # Check if data is available from Arduino
        if arduino and arduino.in_waiting > 0:
            raw_data = arduino.readline().decode('utf-8').strip()  # Read and decode the data
            sensor_data = parse_data(raw_data)  # Parse the data into a dictionary

            # Extract the sensor values
            water_level_upper = sensor_data.get('WaterLevelRight', 0)
            water_level_lower = sensor_data.get('WaterLevelLeft', 0)
            pH = sensor_data.get('pH', 0)
            oxygen = sensor_data.get('Oxygen', 0)
            temperature = sensor_data.get('Temperature', 0)
            light = sensor_data.get('Light', 0)

            # Print the data being sent for logging purposes
            print(f"Sending data - Upper Level: {water_level_upper:.2f} cm, Lower Level: {water_level_lower:.2f} cm, pH: {pH:.2f}, Oxygen: {oxygen:.2f} mg/L, Temperature: {temperature:.2f}°C, Light: {light:.2f}")
                
             # Send the generated data to Adafruit
            aio.send(pH_Feed.key, pH)
            aio.send(oxy_Feed.key, oxygen)
            aio.send(temp_Feed.key, temperature)
            aio.send(light_Feed.key, light)
            aio.send(botWater_Feed.key, water_level_lower)
            aio.send(topWater_Feed.key, water_level_upper)
            aio.send(camera_Feed.key, plantHealth)
            
            if lcd:
                lcd.clear()
                lcd.write_string('==Aquarium Values:==')
                lcd.crlf()
                lcd.write_string(f'Oxygen: {oxygen} mg/L')
                lcd.crlf()
                lcd.write_string(f'Temperature: {temperature} C')
                lcd.crlf()
                lcd.write_string(f'pH: {pH}')
                time.sleep(5)
                
                lcd.clear()
                lcd.write_string('====Pipe Values:====')
                lcd.crlf()
                lcd.write_string(f'Upper: {water_level_upper} cm')
                lcd.crlf()
                lcd.write_string(f'Lower: {water_level_lower} cm')
                lcd.crlf()
                time.sleep(5)
                
                lcd.clear()
                lcd.write_string('===Plant Health:===')
                lcd.crlf()
                lcd.write_string(f'{plantHealth}')
                lcd.crlf()
                time.sleep(5)
                
                if water_level_upper < 4:
                        buzzer.on()
                        time.sleep(1)
                        buzzer.off()
                        time.sleep(1)
                        lcd.clear()
                        lcd.write_string('Please check the pipe!')
                        lcd.crlf()
                        
                if water_level_lower < 1:
                        buzzer.on()
                        time.sleep(1)
                        buzzer.off()
                        time.sleep(1)
                        lcd.write_string('Please check the pipe!')
                        lcd.crlf()
                    
            # avoid timeout from adafruit io
        time.sleep(15)
            
except KeyboardInterrupt:
    # Handle keyboard interrupt (Ctrl+C) to exit the program gracefully
    print("Program terminated by user.")
finally:
    if arduino:
        arduino.close()
    if picam2:
        picam2.stop()
