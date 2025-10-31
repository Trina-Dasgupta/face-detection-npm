import { FaceLandmarkDetector } from './FaceLandmarkDetector.js';

export class FaceVisibilityChecker {
  checkFaceVisibility(facePrediction) {
    const landmarks = facePrediction && facePrediction.scaledMesh;
    if (!landmarks || !Array.isArray(landmarks) || landmarks.length === 0) {
      return {
        isVisible: false,
        score: 0,
        message: 'No facial landmarks available',
        details: {}
      };
    }

    const regions = FaceLandmarkDetector.getLandmarkRegions(landmarks);
    const faceSize = FaceLandmarkDetector.calculateFaceSize(landmarks);

    let score = 100;
    const issues = [];

    // Normalize checks by face height (pixel -> ratio) to make thresholds consistent
    const faceHeight = faceSize.height || (Math.sqrt(faceSize.area) || 1);

    // Check if eyes are open (expected eye height ratio ~0.06 of face height)
    const eyeOpenScore = this.checkEyesOpen(regions, faceHeight);
    if (eyeOpenScore < 0.5) { // More permissive eye threshold
      score -= 15;
      issues.push('Eyes not clearly visible or closed');
    }

    // Check if mouth is closed (expected mouth height ratio when closed ~0.01-0.02)
    const mouthOpenScore = this.checkMouthOpen(regions, faceHeight);
    if (mouthOpenScore > 0.4) { // Allow slightly more mouth variation
      score -= 10;
      issues.push('Mouth should be closed');
    }

    // Check face orientation (simple frontality check)
    const orientationScore = this.checkFaceOrientation(regions);
    if (orientationScore < 0.6) { // Allow more head angle variation
      score -= 15;
      issues.push('Face not facing forward');
    }

    // Check for obstructions (sunglasses, masks, etc.)
    const obstructionScore = this.checkObstructions(regions, faceHeight);
    if (obstructionScore < 0.75) { // More permissive obstruction check
      score -= 20;
      issues.push('Face obstructed (sunglasses, mask, etc.)');
    }

    // Check face size (too small or too large)
    const sizeScore = this.checkFaceSize(faceSize);
    if (sizeScore < 0.6) { // Allow wider range of face sizes
      score -= 10;
      issues.push('Face size inappropriate - move closer or further');
    }

    // Check lighting conditions
    const lightingScore = this.checkLighting(facePrediction);
    if (lightingScore < 0.5) { // More permissive lighting check
      score -= 5;
      issues.push('Poor lighting conditions');
    }

    // Debug: log all scores to help tune thresholds
    console.log('Face visibility scores:', {
      total: score,
      eyes: eyeOpenScore,
      mouth: mouthOpenScore,
      orientation: orientationScore,
      obstruction: obstructionScore,
      size: sizeScore,
      lighting: lightingScore,
      height: faceHeight
    });

    // Relax the overall threshold and boost scores
    const finalScore = Math.min(100, score + 15); // Give a small boost
    const isVisible = finalScore >= 65; // Lower threshold

    return {
      isVisible,
      score: Math.max(0, finalScore),
      message: issues.length > 0 ? issues.join('; ') : 'Face properly visible',
      details: {
        eyeOpenScore,
        mouthOpenScore,
        orientationScore,
        obstructionScore,
        sizeScore,
        lightingScore
      }
    };
  }

  checkEyesOpen(regions, faceHeight) {
    const leftEyeHeight = this.calculateRegionHeight(regions.leftEye);
    const rightEyeHeight = this.calculateRegionHeight(regions.rightEye);

    // Normalize by face height to get a ratio
    const normalizedHeight = ((leftEyeHeight + rightEyeHeight) / 2) / (faceHeight || 1);
    // Typical open eye height ratio ~0.06
    const expectedOpenRatio = 0.06;
    return Math.min(1, normalizedHeight / expectedOpenRatio);
  }

  checkMouthOpen(regions, faceHeight) {
    const mouthHeight = this.calculateRegionHeight(regions.lipsOuter);
    const normalized = (mouthHeight || 0) / (faceHeight || 1);
    // Typical mouth open ratio threshold ~0.04 (4% of face height)
    const openRatio = 0.04;
    return Math.min(1, normalized / openRatio);
  }

  checkFaceOrientation(regions) {
    // Simple frontality check using symmetry
    const leftEyeCenter = this.calculateRegionCenter(regions.leftEye);
    const rightEyeCenter = this.calculateRegionCenter(regions.rightEye);
    const noseTip = regions.noseTip && regions.noseTip[0];

    if (!leftEyeCenter || !rightEyeCenter || !noseTip) return 0;

    const eyeDistance = Math.abs(leftEyeCenter[0] - rightEyeCenter[0]) || 1;
    const leftNoseDistance = Math.abs(leftEyeCenter[0] - noseTip[0]);
    const rightNoseDistance = Math.abs(rightEyeCenter[0] - noseTip[0]);

    const symmetry = 1 - Math.abs(leftNoseDistance - rightNoseDistance) / eyeDistance;
    return Math.max(0, Math.min(1, symmetry));
  }

  checkObstructions(regions, faceHeight) {
    // Check if key facial features are detectable
    let detectableFeatures = 0;
    const totalFeatures = 4; // eyes, nose, mouth

    if (this.isRegionDetectable(regions.leftEye, faceHeight)) detectableFeatures++;
    if (this.isRegionDetectable(regions.rightEye, faceHeight)) detectableFeatures++;
    if (this.isRegionDetectable(regions.noseTip, faceHeight)) detectableFeatures++;
    if (this.isRegionDetectable(regions.lipsOuter, faceHeight)) detectableFeatures++;

    return detectableFeatures / totalFeatures;
  }

  checkFaceSize(faceSize) {
    // Ideal face area range (normalized)
    const minArea = 0.02;
    const maxArea = 0.3;

    if (faceSize.area < minArea) return faceSize.area / minArea;
    if (faceSize.area > maxArea) return maxArea / faceSize.area;
    return 1;
  }

  checkLighting(facePrediction) {
    // Simple lighting check based on confidence scores
    return facePrediction.faceInViewConfidence || 0.5;
  }

  calculateRegionHeight(region) {
    if (!region || region.length === 0) return 0;
    let minY = Infinity, maxY = -Infinity;
    region.forEach(point => {
      const y = Array.isArray(point) ? point[1] : (point.y ?? 0);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    });
    return maxY - minY;
  }

  calculateRegionCenter(region) {
    if (!region || region.length === 0) return null;
    const sum = region.reduce((acc, point) => {
      const x = Array.isArray(point) ? point[0] : (point.x ?? 0);
      const y = Array.isArray(point) ? point[1] : (point.y ?? 0);
      return [acc[0] + x, acc[1] + y];
    }, [0, 0]);

    return [sum[0] / region.length, sum[1] / region.length];
  }

  isRegionDetectable(region, faceHeight) {
    // Check if region points are valid and not collapsed; compare against faceHeight
    if (!region || !Array.isArray(region) || region.length === 0) return false;
    const height = this.calculateRegionHeight(region);
    const width = this.calculateRegionWidth(region);
    // require region dimensions to be at least a small fraction of face height (e.g., 1%)
    const minRatio = 0.01;
    return (height / (faceHeight || 1)) > minRatio && (width / (faceHeight || 1)) > minRatio;
  }

  calculateRegionWidth(region) {
    if (!region || region.length === 0) return 0;
    let minX = Infinity, maxX = -Infinity;
    region.forEach(point => {
      const x = Array.isArray(point) ? point[0] : (point.x ?? 0);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
    });
    return maxX - minX;
  }
}