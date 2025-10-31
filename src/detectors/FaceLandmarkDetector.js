export class FaceLandmarkDetector {
  static getLandmarkRegions(landmarks) {
    // Face landmark indices based on MediaPipe Face Mesh
    return {
      // Face oval (contour)
      faceOval: landmarks.slice(0, 18),
      
      // Left eyebrow
      leftEyebrow: landmarks.slice(27, 36),
      
      // Right eyebrow
      rightEyebrow: landmarks.slice(17, 27),
      
      // Left eye
      leftEye: landmarks.slice(33, 42),
      leftEyeUpper: landmarks.slice(33, 37),
      leftEyeLower: landmarks.slice(37, 42),
      
      // Right eye
      rightEye: landmarks.slice(42, 51),
      rightEyeUpper: landmarks.slice(42, 46),
      rightEyeLower: landmarks.slice(46, 51),
      
      // Lips
      lipsOuter: landmarks.slice(48, 60),
      lipsInner: landmarks.slice(60, 68),
      
      // Nose
      noseTip: landmarks.slice(51, 55),
      noseBridge: landmarks.slice(55, 60)
    };
  }

  static calculateFaceSize(landmarks) {
    const faceOval = landmarks.slice(0, 18);
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    faceOval.forEach(point => {
      minX = Math.min(minX, point[0]);
      maxX = Math.max(maxX, point[0]);
      minY = Math.min(minY, point[1]);
      maxY = Math.max(maxY, point[1]);
    });

    return {
      width: maxX - minX,
      height: maxY - minY,
      area: (maxX - minX) * (maxY - minY)
    };
  }
}