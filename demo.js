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

  // 🟢 Create joints (spheres)
  const jointMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
  const joints = [];
  for (let i = 0; i < 17; i++) {
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.05), jointMaterial);
    scene.add(sphere);
    joints.push(sphere);
  }

  // 🟦 Define skeleton connections (bones)
  const connections = [
    [5, 7], [7, 9],     // left arm
    [6, 8], [8, 10],    // right arm
    [5, 6],             // shoulders
    [11, 13], [13, 15], // left leg
    [12, 14], [14, 16], // right leg
    [11, 12],           // hips
    [5, 11], [6, 12]    // torso sides
  ];

  // 🟨 Create bones (cylinders)
  const boneMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 });
  const bones = connections.map(() => {
    const geometry = new THREE.CylinderGeometry(0.02, 0.02, 1, 8);
    const bone = new THREE.Mesh(geometry, boneMaterial);
    scene.add(bone);
    return bone;
  });

  // 🤖 Load MoveNet detector
  const detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
  );

  // 🎮 Update bone position between two joints
  function updateBone(bone, start, end) {
    const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
    bone.position.copy(mid);

    const direction = new THREE.Vector3().subVectors(end, start);
    const length = direction.length();
    bone.scale.set(1, length, 1);

    const axis = new THREE.Vector3(0, 1, 0);
    const quaternion = new THREE.Quaternion().setFromUnitVectors(
      axis.clone().normalize(),
      direction.clone().normalize()
    );
    bone.setRotationFromQuaternion(quaternion);
  }

  // 🎮 Animation loop
  async function animate() {
    requestAnimationFrame(animate);

    const poses = await detector.estimatePoses(video);
    if (poses.length > 0) {
      const keypoints = poses[0].keypoints;

      keypoints.forEach((kp, i) => {
        if (kp.score > 0.5) {
          const x = (kp.x / video.videoWidth) * 4 - 2;
          const y = -(kp.y / video.videoHeight) * 4 + 2;
          joints[i].position.set(x, y, 0);
        }
      });

      connections.forEach(([a, b], i) => {
        const kpA = keypoints[a];
        const kpB = keypoints[b];
        if (kpA.score > 0.5 && kpB.score > 0.5) {
          const start = joints[a].position;
          const end = joints[b].position;
          updateBone(bones[i], start, end);
        }
      });
    }

    renderer.render(scene, camera);
  }

  animate();
}

main();
