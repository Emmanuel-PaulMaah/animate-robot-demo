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
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);
  camera.position.set(0, 0, 5);

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const light = new THREE.PointLight(0xffffff, 1);
  light.position.set(5, 5, 5);
  scene.add(light);

  // Floor grid
  const grid = new THREE.GridHelper(10, 20);
  grid.position.y = -2;
  scene.add(grid);

  // 🟢 Create joints (spheres)
  const jointMaterial = new THREE.MeshLambertMaterial({ color: 0x00ff00 });
  const joints = [];
  for (let i = 0; i < 17; i++) {
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.08), jointMaterial);
    scene.add(sphere);
    joints.push(sphere);
  }

  // 🟦 Define skeleton connections
  const connections = [
    [5, 7], [7, 9],     // left arm
    [6, 8], [8, 10],    // right arm
    [5, 6],             // shoulders
    [11, 13], [13, 15], // left leg
    [12, 14], [14, 16], // right leg
    [11, 12],           // hips
    [5, 11], [6, 12]    // torso
  ];

  // 🟨 Create bones (cylinders)
  const boneMaterial = new THREE.MeshLambertMaterial({ color: 0xffff00 });
  const bones = connections.map(() => {
    const geometry = new THREE.CylinderGeometry(0.04, 0.04, 1, 8);
    const bone = new THREE.Mesh(geometry, boneMaterial);
    scene.add(bone);
    return bone;
  });

  // 🤖 Load MoveNet detector
  const detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
  );

  // Store smoothed joint positions
  const smoothed = Array(17).fill().map(() => new THREE.Vector3());

  // 🎮 Update bone helper
  function updateBone(bone, start, end) {
    const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
    bone.position.copy(mid);

    const direction = new THREE.Vector3().subVectors(end, start);
    const length = direction.length();
    bone.scale.set(1, length, 1);

    bone.rotation.set(0, 0, 0);
    bone.lookAt(end);
    bone.rotateX(Math.PI / 2);
  }

  // 📡 Run pose detection at ~30fps
  setInterval(async () => {
    const poses = await detector.estimatePoses(video);
    if (poses.length > 0) {
      const keypoints = poses[0].keypoints;

      keypoints.forEach((kp, i) => {
        if (kp.score > 0.5) {
          let x = (kp.x / video.videoWidth) * 4 - 2;
          let y = -(kp.y / video.videoHeight) * 4 + 2;

          x = -x;      // Mirror
          x -= 1.5;    // Shift left

          // Smooth with lerp (0.3 factor)
          smoothed[i].lerp(new THREE.Vector3(x, y, 0), 0.3);
        }
      });
    }
  }, 1000 / 30); // 30fps

  // 🎮 Render loop (decoupled from detection)
  function renderLoop() {
    requestAnimationFrame(renderLoop);

    // Update joints & bones from smoothed positions
    joints.forEach((joint, i) => joint.position.copy(smoothed[i]));
    connections.forEach(([a, b], i) => updateBone(bones[i], smoothed[a], smoothed[b]));

    renderer.render(scene, camera);
  }

  renderLoop();
}

main();
