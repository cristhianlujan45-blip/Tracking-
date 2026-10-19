// three.js viewer: loads a GLB, frames it, and offers wireframe / auto-rotate toggles.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x14161d);
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(3, 5, 4);
    this.scene.add(sun, new THREE.HemisphereLight(0xffffff, 0x222233, 0.6));

    this.grid = new THREE.GridHelper(10, 20, 0x3a3f50, 0x262a36);
    this.scene.add(this.grid);

    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
    this.camera.position.set(2.5, 2, 3);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 1.5;

    this.loader = new GLTFLoader();
    this.model = null;
    this.wireframe = false;

    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.canvas.parentElement;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  async load(url) {
    const gltf = await this.loader.loadAsync(url);
    this.clear();
    this.model = gltf.scene;

    // center on the origin and sit on the grid
    const box = new THREE.Box3().setFromObject(this.model);
    const center = box.getCenter(new THREE.Vector3());
    this.model.position.sub(new THREE.Vector3(center.x, box.min.y, center.z));
    this.scene.add(this.model);
    this.setWireframe(this.wireframe);
    this.frame();
    return this.stats();
  }

  clear() {
    if (!this.model) return;
    this.scene.remove(this.model);
    this.model.traverse((o) => {
      o.geometry?.dispose();
      [].concat(o.material || []).forEach((m) => {
        Object.values(m).forEach((v) => v?.isTexture && v.dispose());
        m.dispose();
      });
    });
    this.model = null;
  }

  frame() {
    if (!this.model) return;
    const box = new THREE.Box3().setFromObject(this.model);
    const size = box.getSize(new THREE.Vector3()).length();
    const center = box.getCenter(new THREE.Vector3());
    const dist = size / (2 * Math.tan((this.camera.fov * Math.PI) / 360)) * 1.1;
    this.camera.position.copy(center).add(new THREE.Vector3(0.6, 0.45, 0.75).normalize().multiplyScalar(dist));
    this.camera.near = size / 100;
    this.camera.far = size * 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.grid.scale.setScalar(Math.max(size / 3, 0.2));
  }

  setWireframe(on) {
    this.wireframe = on;
    this.model?.traverse((o) => [].concat(o.material || []).forEach((m) => (m.wireframe = on)));
  }

  setAutoRotate(on) {
    this.controls.autoRotate = on;
  }

  stats() {
    let triangles = 0, vertices = 0, meshes = 0;
    this.model?.traverse((o) => {
      if (!o.isMesh) return;
      meshes++;
      const g = o.geometry;
      vertices += g.attributes.position.count;
      triangles += g.index ? g.index.count / 3 : g.attributes.position.count / 3;
    });
    return { triangles, vertices, meshes };
  }
}
