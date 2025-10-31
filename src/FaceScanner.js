import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import '@tensorflow/tfjs-backend-webgl';
import { FaceVisibilityChecker } from './detectors/FaceVisibilityChecker.js';
import { ImageUtils } from './utils/ImageUtils.js';

export class FaceScanner {
  constructor(options = {}) {
    this.options = {
      minFaceSize: 0.1,
      maxFaces: 1,
      detectionConfidence: 0.8,
      landmarkConfidence: 0.5,
      autoCaptureDelay: 1000,
      // Do not reference node 'path' module at construction time (browser builds).
      // Python is optional and enabled via `enablePython: true` when running in Node.
      enablePython: false,
      pythonPath: null,
      ...options
    };
    
    this.model = null;
    this.analyzer = null;
    this.isInitialized = false;
    this.visibilityChecker = new FaceVisibilityChecker();
    this.captureTimeout = null;
  }

  // Ensure the prediction uses a consistent format: provide scaledMesh (array of [x,y]) and boundingBox
  _normalizePrediction(pred) {
    if (!pred) return pred;

    const normalized = Object.assign({}, pred);

    // Prefer scaledMesh, then mesh, then keypoints
    if (Array.isArray(pred.scaledMesh) && pred.scaledMesh.length > 0) {
      normalized.scaledMesh = pred.scaledMesh;
    } else if (Array.isArray(pred.mesh) && pred.mesh.length > 0) {
      normalized.scaledMesh = pred.mesh;
    } else if (Array.isArray(pred.keypoints) && pred.keypoints.length > 0) {
      // keypoints can be array of {x,y,z} or [x,y,z]
      normalized.scaledMesh = pred.keypoints.map(k => {
        if (Array.isArray(k)) return [k[0], k[1]];
        return [k.x !== undefined ? k.x : k[0], k.y !== undefined ? k.y : k[1]];
      });
    } else {
      // No recognizable landmarks — leave as-is (visibility checker will handle missing data)
      normalized.scaledMesh = normalized.scaledMesh || null;
    }

    // Ensure boundingBox exists: use model-provided or compute from scaledMesh if available
    if (!normalized.boundingBox || !normalized.boundingBox.topLeft) {
      if (Array.isArray(normalized.scaledMesh) && normalized.scaledMesh.length > 0) {
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        normalized.scaledMesh.forEach(pt => {
          const x = pt[0];
          const y = pt[1];
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        });
        normalized.boundingBox = {
          topLeft: [minX, minY],
          bottomRight: [maxX, maxY]
        };
      }
    }

    return normalized;
  }

  async initialize() {
    try {
      console.log('Loading TensorFlow.js model...');
      
      // Use the SupportedModels enum (correct API).
      // The v1+ face-landmarks-detection package exposes createDetector instead
      // of the older `load` helper. Use createDetector and provide model options.
      // Provide a runtime value (required) — default to 'tfjs' since we import the tfjs backend.
      // Allow caller to override via options.runtime (either 'tfjs' or 'mediapipe').
      this.model = await faceLandmarksDetection.createDetector(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
        {
          runtime: this.options.runtime || 'tfjs',
          maxFaces: this.options.maxFaces,
          // refineLandmarks improves face mesh quality (optional)
          refineLandmarks: this.options.refineLandmarks !== undefined ? this.options.refineLandmarks : true,
          // keep confidence values on the options object for compatibility; some runtimes may use them
          detectionConfidence: this.options.detectionConfidence,
          landmarkConfidence: this.options.landmarkConfidence
        }
      );
      
      // Initialize Python analyzer only when running in Node and when explicitly enabled.
      // Avoid bundling python-shell into the browser build (python-shell uses node core modules).
      if (this.options.enablePython && typeof window === 'undefined') {
        try {
          // Dynamically require at runtime using an indirect require to avoid static bundling by webpack
          const PythonShell = eval('require')('python-shell').PythonShell;
          const path = eval('require')('path');
          this.analyzer = new PythonShell(path.join(__dirname, 'python', 'analyzer.py'), {
            pythonPath: this.options.pythonPath,
            mode: 'json'
          });
        } catch (err) {
          console.warn('Python analyzer not available in this environment:', err && err.message);
          this.analyzer = null;
        }
      }

      this.isInitialized = true;
      console.log('Face scanner initialized successfully');
      return true;
    } catch (error) {
      console.error('Failed to initialize face scanner:', error);
      throw error;
    }
  }

  async scanFace(videoElement) {
    if (!this.isInitialized) {
      throw new Error('Face scanner not initialized. Call initialize() first.');
    }

    try {
      // The detector expects an HTMLVideoElement / HTMLImageElement / HTMLCanvasElement
      // or ImageData. Some environments or callers may pass a plain object (e.g. a
      // MediaStream or a wrapper). Try the direct call first, and if TFJS complains
      // about the input type, fall back to drawing a frame to a temporary canvas and
      // passing that canvas to the detector.
      let predictions;
      // The new detector API typically accepts the image element directly: estimateFaces(videoElement)
      // Older code used estimateFaces({input: videoElement, ...}). Try the element-first signature first,
      // then fall back to the object signature, and finally to the canvas snapshot approach if TFJS complains.
      try {
        // Preferred: pass the element/canvas directly
        predictions = await this.model.estimateFaces(videoElement);
      } catch (firstErr) {
        // Try the older/object-based signature for backward compatibility
        try {
          predictions = await this.model.estimateFaces({
            input: videoElement,
            returnTensors: false,
            flipHorizontal: false,
            predictIrises: true
          });
        } catch (secondErr) {
          const msg = (secondErr && secondErr.message) || (firstErr && firstErr.message) || '';
          if (msg.includes('pixels passed to tf.browser.fromPixels') || msg.includes('was Object')) {
            // Create a temporary canvas and draw the frame, then retry with that canvas
            const canvas = document.createElement('canvas');
            try {
              canvas.width = (videoElement && (videoElement.videoWidth || videoElement.width)) || 640;
              canvas.height = (videoElement && (videoElement.videoHeight || videoElement.height)) || 480;
              const ctx = canvas.getContext('2d');
              ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

              // Try element-style call with canvas
              try {
                predictions = await this.model.estimateFaces(canvas);
              } catch (thirdErr) {
                // Try object-style call with canvas
                predictions = await this.model.estimateFaces({
                  input: canvas,
                  returnTensors: false,
                  flipHorizontal: false,
                  predictIrises: true
                });
              }
            } catch (drawError) {
              // If drawing fails, throw the most relevant error
              throw secondErr;
            }
          } else {
            // Unknown error — rethrow the latest
            throw secondErr;
          }
        }
      }

      if (predictions.length === 0) {
        return {
          faceDetected: false,
          message: 'No face detected',
          landmarks: null,
          base64Image: null
        };
      }

      // Normalize prediction to ensure downstream code has a consistent shape
      let face = predictions[0];
      face = this._normalizePrediction(face);

      const visibilityResult = this.visibilityChecker.checkFaceVisibility(face);

      if (!visibilityResult.isVisible) {
        return {
          faceDetected: false,
          message: visibilityResult.message,
          landmarks: face.scaledMesh,
          base64Image: null
        };
      }

      // Face is properly visible, auto-capture
      const base64Image = await ImageUtils.captureFrame(videoElement);
      
      return {
        faceDetected: true,
        message: 'Face properly detected and captured',
        landmarks: face.scaledMesh,
        base64Image: base64Image,
        boundingBox: face.boundingBox,
        visibilityScore: visibilityResult.score
      };

    } catch (error) {
      console.error('Error during face scanning:', error);
      throw error;
    }
  }

  async analyzeFace(base64Image) {
    return new Promise((resolve, reject) => {
        this.analyzer.send({ image: base64Image });
        this.analyzer.once('message', resolve);
        this.analyzer.once('error', reject);
    });
  }

  async scanFaceContinuously(videoElement, callback, interval = 500) {
    if (!this.isInitialized) {
      throw new Error('Face scanner not initialized. Call initialize() first.');
    }

    const scanInterval = setInterval(async () => {
      try {
        const result = await this.scanFace(videoElement);
        callback(result);
      } catch (error) {
        callback({
          faceDetected: false,
          message: `Scan error: ${error.message}`,
          landmarks: null,
          base64Image: null
        });
      }
    }, interval);

    return () => clearInterval(scanInterval);
  }

  dispose() {
    if (this.model) {
      // New detector API may provide either dispose() or close(); handle both.
      if (typeof this.model.dispose === 'function') {
        this.model.dispose();
      } else if (typeof this.model.close === 'function') {
        this.model.close();
      }
      this.model = null;
    }
    if (this.analyzer) {
      this.analyzer.end();
      this.analyzer = null;
    }
    this.isInitialized = false;
  }
}