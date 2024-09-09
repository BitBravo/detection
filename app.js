const tf = require('@tensorflow/tfjs-node');
const NodeWebcam = require('node-webcam');
const { createCanvas, Image } = require('canvas');
const path = require('path');

// Webcam configuration
const webcamOptions = {
    width: 640,
    height: 480,
    quality: 100,
    frames: 30,
    saveShots: false,
    device: false,
    callbackReturn: 'buffer',
    verbose: false
};

const Webcam = NodeWebcam.create(webcamOptions);

// Load the pre-trained model
const modelPath = path.join(__dirname, 'model', 'ssdlite_mobilenet_v2', 'web_model');
let model;

// Function to load the model
async function loadModel() {
    model = await tf.loadGraphModel(`file://${modelPath}/model.json`);
}


async function detect(frameBuffer) {
    const image = new Image();
    image.src = frameBuffer;
  
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, image.width, image.height);
  
    const inputTensor = tf.browser.fromPixels(canvas);
    const expandedTensor = inputTensor.expandDims(0);
    const detections = await model.executeAsync(expandedTensor);
  
    return detections;
  }
  
  // Process frames from webcam
  async function processFrames() {
    await loadModel();
    console.log('Model loaded successfully');
  
    setInterval(() => {
      Webcam.capture('frame', async (err, frameBuffer) => {
        if (err) {
          console.error('Error capturing frame:', err);
          return;
        }
  
        const detections = await detect(frameBuffer);
  
        // Extract detection information
        const detectionBoxes = detections[1].arraySync();
        detectionBoxes.forEach(box => {
          const [ymin, xmin, ymax, xmax] = box;
          const left = xmin * webcamOptions.width;
          const right = xmax * webcamOptions.width;
          const top = ymin * webcamOptions.height;
          const bottom = ymax * webcamOptions.height;
  
          console.log(`Detected human at: left=${left}, top=${top}, right=${right}, bottom=${bottom}`);
        });
      });
    }, 1000 / webcamOptions.frames);
  }
  
  processFrames();