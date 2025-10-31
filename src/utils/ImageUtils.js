class ImageUtils {
  static async captureFrame(videoElement) {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      
      canvas.width = videoElement.videoWidth;
      canvas.height = videoElement.videoHeight;
      
      context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
      
      const base64Image = canvas.toDataURL('image/jpeg', 0.9);
      resolve(base64Image);
    });
  }

  static drawLandmarks(canvas, landmarks, options = {}) {
    const ctx = canvas.getContext('2d');
    const { 
      pointColor = 'red', 
      pointSize = 2,
      lineColor = 'blue',
      lineWidth = 1 
    } = options;

    ctx.strokeStyle = lineColor;
    ctx.fillStyle = pointColor;
    ctx.lineWidth = lineWidth;

    landmarks.forEach(point => {
      ctx.beginPath();
      ctx.arc(point[0], point[1], pointSize, 0, 2 * Math.PI);
      ctx.fill();
    });

    if (landmarks.length >= 18) {
      ctx.beginPath();
      ctx.moveTo(landmarks[0][0], landmarks[0][1]);
      for (let i = 1; i < 18; i++) {
        ctx.lineTo(landmarks[i][0], landmarks[i][1]);
      }
      ctx.closePath();
      ctx.stroke();
    }
  }
}

module.exports = { ImageUtils };