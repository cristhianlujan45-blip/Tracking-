// three.js viewer: loads a GLB, frames it, and offers view modes, lighting presets,
// animation playback, PNG snapshots and 360° turntable video recording.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const LIGHTING = {
  studio: { bg: 0x14161d, env: 1.0, sun: 1.2, sunColor: 0xffffff, hemi: 0.6, grid: [0x3a3f50, 0x262a36] },
  outdoor: { bg: 0x9bbfe0, env: 0.8, sun: 2.4, sunColor: 0xfff1d6, hemi: 1.0, grid: [0x6f8fae, 0x86a7c6] },
  night: { bg: 0x05060a, env: 0.25, sun: 0.5, sunColor: 0x8fa8ff, hemi: 0.15, grid: [0x1c2030, 0x11131c] },
};

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;

    this.scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    this.sun = new THREE.DirectionalLight(0xffffff, 1.2);
    this.sun.position.set(3, 5, 4);
    this.hemi = new THREE.HemisphereLight(0xffffff, 0x222233, 0.6);
    this.rim = new THREE.DirectionalLight(0x6c8cff, 0);
    this.rim.position.set(-4, 2, -4);
    this.scene.add(this.sun, this.hemi, this.rim);

    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
    this.camera.position.set(2.5, 2, 3);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 1.5;

    this.loader = new GLTFLoader();
    this.clock = new THREE.Clock();
    this.model = null;
    this.mixer = null;
    this.clips = [];
    this.action = null;
    this.viewMode = "textured";
    this.overrides = {
      clay: new THREE.MeshStandardMaterial({ color: 0xd9d3cc, roughness: 0.85, metalness: 0 }),
      normals: new THREE.MeshNormalMaterial(),
    };
    this.turntable = null;
    this.gridVisible = true;
    this.setLighting("studio");

    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  tick() {
    const dt = this.clock.getDelta();
    this.mixer?.update(dt);
    if (this.turntable && this.model) {
      const t = Math.min((performance.now() - this.turntable.start) / this.turntable.duration, 1);
      this.model.rotation.y = this.turntable.rotation + t * Math.PI * 2;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
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
    this.model.traverse((o) => {
      if (!o.isMesh) return;
      o.userData.original = o.material;
      // override materials (clay, normals) need normals even when the file has none
      if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
    });
    this.scene.add(this.model);

    this.clips = gltf.animations || [];
    if (this.clips.length) {
      this.mixer = new THREE.AnimationMixer(this.model);
      this.playClip(0);
    }
    this.setViewMode(this.viewMode);
    this.frame();
    return this.stats();
  }

  clear() {
    if (!this.model) return;
    this.mixer?.stopAllAction();
    this.mixer = null;
    this.action = null;
    this.clips = [];
    this.scene.remove(this.model);
    this.model.traverse((o) => {
      o.geometry?.dispose();
      [].concat(o.userData.original || []).forEach((m) => {
        Object.values(m).forEach((v) => v?.isTexture && v.dispose());
        m.dispose();
      });
    });
    this.model = null;
  }

  frame() {
    if (!this.model) return;
    const box = new THREE.Box3().setFromObject(this.model);
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    const center = box.getCenter(new THREE.Vector3());
    const dist = (size / (2 * Math.tan((this.camera.fov * Math.PI) / 360))) * 1.1;
    this.camera.position.copy(center).add(new THREE.Vector3(0.6, 0.45, 0.75).normalize().multiplyScalar(dist));
    this.camera.near = size / 100;
    this.camera.far = size * 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.makeGrid(size);
  }

  makeGrid(size = 3) {
    if (this.grid) {
      this.scene.remove(this.grid);
      this.grid.geometry.dispose();
      this.grid.material.dispose();
    }
    const [a, b] = LIGHTING[this.lighting].grid;
    const extent = Math.max(size * 3, 1);
    this.grid = new THREE.GridHelper(extent, 20, a, b);
    this.grid.visible = this.gridVisible;
    this.scene.add(this.grid);
  }

  // textured | clay | normals | wireframe
  setViewMode(mode) {
    this.viewMode = mode;
    this.model?.traverse((o) => {
      if (!o.isMesh) return;
      const original = o.userData.original;
      o.material = this.overrides[mode] || original;
      [].concat(original).forEach((m) => (m.wireframe = mode === "wireframe"));
    });
  }

  setLighting(preset) {
    const p = LIGHTING[preset] || LIGHTING.studio;
    this.lighting = preset in LIGHTING ? preset : "studio";
    this.scene.background = new THREE.Color(p.bg);
    this.scene.environmentIntensity = p.env;
    this.sun.intensity = p.sun;
    this.sun.color.set(p.sunColor);
    this.hemi.intensity = p.hemi;
    this.rim.intensity = preset === "night" ? 2.0 : 0;
    const size = this.model ? new THREE.Box3().setFromObject(this.model).getSize(new THREE.Vector3()).length() : 3;
    this.makeGrid(size);
  }

  setGrid(on) {
    this.gridVisible = on;
    if (this.grid) this.grid.visible = on;
  }

  setAutoRotate(on) {
    this.controls.autoRotate = on;
  }

  playClip(index) {
    if (!this.mixer || !this.clips[index]) return;
    this.action?.stop();
    this.action = this.mixer.clipAction(this.clips[index]);
    this.action.paused = false;
    this.action.play();
  }

  toggleAnimation() {
    if (!this.action) return false;
    this.action.paused = !this.action.paused;
    return !this.action.paused;
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
    const size = this.model ? new THREE.Box3().setFromObject(this.model).getSize(new THREE.Vector3()) : null;
    return { triangles, vertices, meshes, size, animations: this.clips.map((c, i) => c.name || `Animación ${i + 1}`) };
  }

  // Square JPEG/PNG of the current view, without HUD, for thumbnails and screenshots.
  async snapshot({ size = 0, type = "image/png", hideGrid = false } = {}) {
    const gridWas = this.gridVisible;
    if (hideGrid) this.setGrid(false);
    this.renderer.render(this.scene, this.camera);
    const src = this.canvas;
    let out = src;
    if (size) {
      out = document.createElement("canvas");
      out.width = out.height = size;
      const side = Math.min(src.width, src.height);
      out.getContext("2d").drawImage(src, (src.width - side) / 2, (src.height - side) / 2, side, side, 0, 0, size, size);
    }
    if (hideGrid) this.setGrid(gridWas);
    return new Promise((resolve) => out.toBlob(resolve, type, 0.88));
  }

  // USDZ for Apple AR Quick Look, built in the browser from the loaded model.
  async exportUSDZ() {
    if (!this.model) throw new Error("No hay modelo cargado");
    const { USDZExporter } = await import("three/addons/exporters/USDZExporter.js");
    const mode = this.viewMode;
    this.setViewMode("textured"); // export the real materials, not the clay/normals overrides
    try {
      const data = await new USDZExporter().parseAsync(this.model);
      return new Blob([data], { type: "model/vnd.usdz+zip" });
    } finally {
      this.setViewMode(mode);
    }
  }

  // One full revolution of the model recorded as WebM.
  async recordTurntable(seconds = 6) {
    if (!this.model || !window.MediaRecorder) throw new Error("Grabación no soportada en este navegador");
    const mime = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((m) => MediaRecorder.isTypeSupported(m));
    const stream = this.canvas.captureStream(30);
    const recorder = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 6e6 } : undefined);
    const chunks = [];
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    const autoRotate = this.controls.autoRotate;
    this.controls.autoRotate = false;
    const rotation = this.model.rotation.y;
    this.turntable = { start: performance.now(), duration: seconds * 1000, rotation };
    recorder.start();
    await new Promise((r) => setTimeout(r, seconds * 1000));
    const stopped = new Promise((r) => (recorder.onstop = r));
    recorder.stop();
    await stopped;
    this.turntable = null;
    if (this.model) this.model.rotation.y = rotation;
    this.controls.autoRotate = autoRotate;
    stream.getTracks().forEach((t) => t.stop());
    return new Blob(chunks, { type: "video/webm" });
  }
}
