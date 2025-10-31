const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const { pythonDependencies } = require('../package.json')['face-scanner'];

async function setup() {
    // Check Python installation
    try {
        await runCommand('python', ['-c', 'import sys; print(sys.version)']);
    } catch (err) {
        console.error('Python 3.8+ is required but not found');
        process.exit(1);
    }

    // Create virtual environment
    const venvPath = path.join(__dirname, '..', 'python_env');
    await runCommand('python', ['-m', 'venv', venvPath]);

    // Install dependencies
    const pipCmd = process.platform === 'win32' ? 
        path.join(venvPath, 'Scripts', 'pip.exe') :
        path.join(venvPath, 'bin', 'pip');

    for (const dep of pythonDependencies) {
        await runCommand(pipCmd, ['install', dep]);
    }

    // Download face landmark model
    const modelPath = path.join(__dirname, '..', 'models');
    if (!fs.existsSync(modelPath)) {
        fs.mkdirSync(modelPath);
    }

    // Download dlib face landmark model
    await runCommand(pipCmd, [
        'install',
        'https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2'
    ]);
}

function runCommand(cmd, args) {
    return new Promise((resolve, reject) => {
        const proc = spawn(cmd, args, { stdio: 'inherit' });
        proc.on('close', code => {
            if (code === 0) resolve();
            else reject(new Error(`Command failed with code ${code}`));
        });
    });
}

setup().catch(console.error);