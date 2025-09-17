async function main() {
  const video = document.getElementById("video");

  // 🎥 Setup webcam
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await video.play();

  // 🧠 Setup TensorFlow backend
  await tf.setBackend("webgl");
  await tf.ready();

  // 📦 Setup Three.js scene
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);
  camera.position.z = 5;

  // Add light
  const light = new THREE.PointLight(0xffffff, 1);
  light.position.set(5, 5, 5);
  scene.add(light);

  // 🟢 Create stick-figure joints (17 for MoveNet)
  const jointMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
  const joints = [];
  for (let i = 0; i < 17; i++) {
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.05), jointMaterial);
    scene.add(sphere);
    joints.push(sphere);
  }

  // 🤖 Load MoveNet detector
  const detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
  );

  // 🎮 Animation loop
  async function animate() {
    requestAnimationFrame(animate);

    const poses = await detector.estimatePoses(video);
    if (poses.length > 0) {
      poses[0].keypoints.forEach((kp, i) => {
        if (kp.score > 0.5) {
          // Normalize coordinates to [-2, 2] range
          const x = (kp.x / video.videoWidth) * 4 - 2;
          const y = -(kp.y / video.videoHeight) * 4 + 2;
          joints[i].position.set(x, y, 0);
        }
      });
    }

    renderer.render(scene, camera);
  }

  animate();
}

main();
