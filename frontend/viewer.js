// three.js viewer: loads a GLB, frames it, and offers view modes, lighting presets,
// animation playback, PNG snapshots and 360° turntable video recording.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

// bg: vertical gradient stops (top → bottom); rim: coloured back light for the anime look
const LIGHTING = {
  studio: { bg: ["#1a1f45", "#0d1028", "#06070f"], env: 1.0, sun: 1.2, sunColor: 0xffffff, hemi: 0.6,
            rim: [0xff4f9a, 0.6], grid: [0xff4f9a, 0x2a2f5a] },
  tokyo: { bg: ["#070818", "#2a0c5c", "#ff2e88"], env: 0.7, sun: 0.9, sunColor: 0xd8d0ff, hemi: 0.35,
           rim: [0x3cf2ff, 2.4], grid: [0x3cf2ff, 0x5a1d8f] },
  sunset: { bg: ["#2b1055", "#d7456b", "#ffc371"], env: 0.85, sun: 2.0, sunColor: 0xffc08a, hemi: 0.7,
            rim: [0xff6f61, 1.4], grid: [0xffd23f, 0x8a3060] },
  sakura: { bg: ["#ffe3f0", "#ffc2dc", "#f6a5c9"], env: 1.0, sun: 1.6, sunColor: 0xfff4f8, hemi: 0.9,
            rim: [0xffffff, 0.6], grid: [0xff4f9a, 0xe58fb8] },
  outdoor: { bg: ["#5aa2e8", "#9bc9f0", "#dcebf7"], env: 0.8, sun: 2.4, sunColor: 0xfff1d6, hemi: 1.0,
             rim: [0xffffff, 0], grid: [0x6f8fae, 0x86a7c6] },
  night: { bg: ["#020309", "#070a1c", "#0d1430"], env: 0.25, sun: 0.5, sunColor: 0x8fa8ff, hemi: 0.15,
           rim: [0x6c8cff, 2.0], grid: [0x1c2a50, 0x11131c] },
};

// which channel of each map to show in the material inspector (-1 = all colours)
const MAP_CHANNEL = { map: -1, normalMap: -1, roughnessMap: 1, metalnessMap: 2, aoMap: 0 };

function channelMaterial(texture, channel, isColor) {
  return new THREE.ShaderMaterial({
    uniforms: { map: { value: texture }, channel: { value: channel } },
    defines: isColor ? { IS_COLOR: "" } : {},
    vertexShader: `
      #include <common>
      #include <skinning_pars_vertex>
      varying vec2 vUv;
      void main() {
        vUv = uv;
        #include <skinbase_vertex>
        #include <begin_vertex>
        #include <skinning_vertex>
        #include <project_vertex>
      }`,
    fragmentShader: `
      uniform sampler2D map;
      uniform int channel;
      varying vec2 vUv;
      void main() {
        vec4 c = texture2D(map, vUv);
        gl_FragColor = vec4(channel < 0 ? c.rgb : vec3(c[channel]), 1.0);
        #ifdef IS_COLOR
        #include <colorspace_fragment>
        #endif
      }`,
  });
}

function gradientTexture(stops) {
  const canvas = Object.assign(document.createElement("canvas"), { width: 4, height: 256 });
  const g = canvas.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, 256);
  stops.forEach((c, i) => grad.addColorStop(i / (stops.length - 1), c));
  g.fillStyle = grad;
  g.fillRect(0, 0, 4, 256);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

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
    this.mapView = "";
    this.wireMaterial = new THREE.MeshBasicMaterial({ color: 0xd4ff5a, wireframe: true, transparent: true, opacity: 0.35 });
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
    if (this.editor) this._drawMarkers(); // follow zoom / pan
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
    this.place();
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

  // centre on the origin and sit on the grid, after an optional turn about Y (degrees)
  place(yawDegrees = 0) {
    this.model.rotation.y = THREE.MathUtils.degToRad(yawDegrees);
    this.model.position.set(0, 0, 0);
    this.model.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(this.model);
    const center = box.getCenter(new THREE.Vector3());
    this.model.position.set(-center.x, -box.min.y, -center.z);
    this.model.updateMatrixWorld(true);
  }

  clear() {
    if (!this.model) return;
    this.exitMarkerEditor();
    this.setSkeletonVisible(false);
    this.weightBone = null;
    this.mapView = "";
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
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.55;
    this.grid.visible = this.gridVisible;
    this.scene.add(this.grid);
  }

  // textured | wireover (texture + wireframe on top) | clay | normals | wireframe
  setViewMode(mode) {
    this.viewMode = mode;
    const meshes = [];
    this.model?.traverse((o) => o.isMesh && !o.userData.isOverlay && meshes.push(o));
    for (const o of meshes) {
      const original = o.userData.original;
      o.material = this.overrides[mode] || this._mapMaterial(o) || original;
      [].concat(original).forEach((m) => (m.wireframe = mode === "wireframe"));
      if (mode === "wireover" && !o.userData.overlay) {
        const overlay = o.isSkinnedMesh ? new THREE.SkinnedMesh(o.geometry, this.wireMaterial) : new THREE.Mesh(o.geometry, this.wireMaterial);
        if (o.isSkinnedMesh) overlay.bind(o.skeleton, o.bindMatrix);
        overlay.userData.isOverlay = true;
        overlay.renderOrder = 1;
        o.add(overlay); // same transform as the mesh it outlines
        o.userData.overlay = overlay;
      } else if (mode !== "wireover" && o.userData.overlay) {
        o.remove(o.userData.overlay);
        delete o.userData.overlay;
      }
    }
  }

  // Material inspector: show one map (colour, normal, roughness, metal, AO) instead of the shaded result.
  setMapView(key) {
    this.mapView = key || "";
    this.setViewMode(this.viewMode === "clay" || this.viewMode === "normals" ? "textured" : this.viewMode);
  }

  _mapMaterial(mesh) {
    if (!this.mapView) return null;
    const source = [].concat(mesh.userData.original)[0];
    const texture = source?.[this.mapView];
    if (!texture) return null;
    mesh.userData.mapMaterials ||= {};
    return (mesh.userData.mapMaterials[this.mapView] ||= channelMaterial(texture, MAP_CHANNEL[this.mapView], this.mapView === "map"));
  }

  availableMaps() {
    const found = new Set();
    this.model?.traverse((o) => {
      if (!o.isMesh || o.userData.isOverlay) return;
      [].concat(o.userData.original || o.material).forEach((m) => Object.keys(MAP_CHANNEL).forEach((k) => m?.[k] && found.add(k)));
    });
    return [...found];
  }

  // largest texture side in pixels (0 when the model has no textures)
  textureSize() {
    let size = 0;
    this.model?.traverse((o) => {
      if (!o.isMesh || o.userData.isOverlay) return;
      [].concat(o.userData.original || o.material).forEach((m) => {
        const img = m?.map?.image;
        if (img) size = Math.max(size, img.width || 0, img.height || 0);
      });
    });
    return size;
  }

  setLighting(preset) {
    const p = LIGHTING[preset] || LIGHTING.studio;
    this.lighting = preset in LIGHTING ? preset : "studio";
    this.scene.background?.dispose?.();
    this.scene.background = gradientTexture(p.bg);
    this.scene.environmentIntensity = p.env;
    this.sun.intensity = p.sun;
    this.sun.color.set(p.sunColor);
    this.hemi.intensity = p.hemi;
    this.rim.color.set(p.rim[0]);
    this.rim.intensity = p.rim[1];
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

  hasTextures() {
    let found = false;
    this.model?.traverse((o) => {
      if (o.isMesh) [].concat(o.userData.original || o.material).forEach((m) => (found ||= !!m.map));
    });
    return found;
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

  // ---------------------------------------------------------------- rig inspection

  skinnedMeshes() {
    const out = [];
    this.model?.traverse((o) => o.isSkinnedMesh && !o.userData.isOverlay && out.push(o));
    return out;
  }

  boneNames() {
    const mesh = this.skinnedMeshes()[0];
    return mesh ? mesh.skeleton.bones.map((b) => b.name).filter((n) => !n.endsWith("_End")) : [];
  }

  setSkeletonVisible(on) {
    if (this.skeletonHelper) {
      this.scene.remove(this.skeletonHelper);
      this.skeletonHelper.dispose?.();
      this.skeletonHelper = null;
    }
    if (on && this.model && this.skinnedMeshes().length) {
      this.skeletonHelper = new THREE.SkeletonHelper(this.model);
      this.skeletonHelper.material.depthTest = false;
      this.skeletonHelper.material.linewidth = 2;
      this.scene.add(this.skeletonHelper);
    }
  }

  // Heat map (blue → red) of how strongly each vertex follows one bone; null restores the look.
  setWeightBone(name) {
    this.weightBone = name || null;
    for (const mesh of this.skinnedMeshes()) {
      const geo = mesh.geometry;
      if (!name) {
        if (geo.userData.originalColor !== undefined) {
          if (geo.userData.originalColor) geo.setAttribute("color", geo.userData.originalColor);
          else geo.deleteAttribute("color");
          delete geo.userData.originalColor;
        }
        mesh.material = mesh.userData.weightMaterial ? mesh.userData.original : mesh.material;
        delete mesh.userData.weightMaterial;
        continue;
      }
      const boneIndex = mesh.skeleton.bones.findIndex((b) => b.name === name);
      const idx = geo.attributes.skinIndex, w = geo.attributes.skinWeight;
      const colors = new Float32Array(idx.count * 3);
      const color = new THREE.Color();
      for (let i = 0; i < idx.count; i++) {
        let value = 0;
        for (let k = 0; k < 4; k++) if (idx.getComponent(i, k) === boneIndex) value += w.getComponent(i, k);
        color.setHSL((1 - value) * 0.66, 0.9, 0.35 + 0.2 * value);
        colors.set([color.r, color.g, color.b], i * 3);
      }
      if (geo.userData.originalColor === undefined) geo.userData.originalColor = geo.attributes.color || null;
      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      if (!mesh.userData.weightMaterial) {
        mesh.userData.weightMaterial = new THREE.MeshBasicMaterial({ vertexColors: true });
      }
      mesh.material = mesh.userData.weightMaterial;
    }
    if (!name) this.setViewMode(this.viewMode);
  }

  // ---------------------------------------------------------------- marker editor
  // view "front": markers are [x, y] on the plane z = 0, seen from +Z (humanoids).
  // view "side": markers are [z, y] on the plane x = 0, seen from -X so the head (+Z) is on the right (animals).

  enterMarkerEditor({ markers, labels, yaw = 0, symmetric = true, view = "front", onChange }) {
    this.exitMarkerEditor();
    if (!this.model) return;
    this.place(yaw);
    const box = new THREE.Box3().setFromObject(this.model);
    const size = box.getSize(new THREE.Vector3());
    const h = size.y || 1;
    const side = view === "side";
    this.editor = { markers: structuredClone(markers), symmetric: symmetric && !side, side, onChange,
                    autoRotate: this.controls.autoRotate };
    this.controls.autoRotate = false;
    this.controls.enableRotate = false;
    const width = side ? size.z : size.x;
    const dist = (Math.max(h, width) / (2 * Math.tan((this.camera.fov * Math.PI) / 360))) * 1.15;
    if (side) this.camera.position.set(-(dist + size.x), h / 2, 0);
    else this.camera.position.set(0, h / 2, dist + size.z);
    this.controls.target.set(0, h / 2, 0);
    this.controls.update();

    const wrap = this.canvas.parentElement;
    const layer = document.createElement("div");
    layer.className = "marker-layer";
    layer.innerHTML = '<svg class="marker-bones"></svg>';
    wrap.append(layer);
    this.editor.layer = layer;
    this.editor.dots = {};
    for (const [name, label] of Object.entries(labels)) {
      const dot = document.createElement("div");
      const kind = name.endsWith("_r") || name.startsWith("front_") ? "right"
        : name.endsWith("_l") || name.startsWith("back_") ? "left" : "center";
      dot.className = `marker ${kind}`;
      dot.title = label;
      dot.dataset.label = label;
      dot.addEventListener("pointerdown", (e) => this._dragMarker(e, name));
      layer.append(dot);
      this.editor.dots[name] = dot;
    }
    this._drawMarkers();
  }

  exitMarkerEditor() {
    if (!this.editor) return;
    this.editor.layer.remove();
    this.controls.enableRotate = true;
    this.controls.autoRotate = this.editor.autoRotate;
    this.editor = null;
    if (this.model) {
      this.place(0);
      this.frame();
    }
  }

  setMarkers(markers) {
    if (!this.editor) return;
    this.editor.markers = structuredClone(markers);
    this._drawMarkers();
  }

  setMarkerSymmetry(on) {
    if (this.editor) this.editor.symmetric = on && !this.editor.side;
  }

  // marker coordinates (a, b) → canvas pixels
  _toScreen(a, b) {
    const v = (this.editor?.side ? new THREE.Vector3(0, b, a) : new THREE.Vector3(a, b, 0)).project(this.camera);
    const { clientWidth: w, clientHeight: h } = this.canvas;
    return [((v.x + 1) / 2) * w, ((1 - v.y) / 2) * h];
  }

  _toWorld(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    const hit = new THREE.Vector3();
    const normal = this.editor?.side ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, 1);
    return ray.ray.intersectPlane(new THREE.Plane(normal, 0), hit) ? hit : null;
  }

  _dragMarker(event, name) {
    event.preventDefault();
    const dot = this.editor.dots[name];
    dot.setPointerCapture(event.pointerId);
    const move = (e) => {
      const p = this._toWorld(e.clientX, e.clientY);
      if (!p || !this.editor) return;
      const m = this.editor.markers;
      if (this.editor.side) {
        m[name] = [+p.z.toFixed(4), +p.y.toFixed(4)];
        this._drawMarkers();
        this.editor.onChange?.(m);
        return;
      }
      const centre = name === "chin" || name === "groin";
      m[name] = [centre ? 0 : +p.x.toFixed(4), +p.y.toFixed(4)];
      if (this.editor.symmetric && !centre) {
        const other = name.endsWith("_l") ? name.replace(/_l$/, "_r") : name.replace(/_r$/, "_l");
        m[other] = [-m[name][0], m[name][1]];
      }
      this._drawMarkers();
      this.editor.onChange?.(m);
    };
    const up = () => {
      dot.removeEventListener("pointermove", move);
      dot.removeEventListener("pointerup", up);
    };
    dot.addEventListener("pointermove", move);
    dot.addEventListener("pointerup", up);
  }

  _drawMarkers() {
    const ed = this.editor;
    if (!ed) return;
    const m = ed.markers;
    const pt = (name) => this._toScreen(m[name][0], m[name][1]);
    for (const [name, dot] of Object.entries(ed.dots)) {
      const [x, y] = pt(name);
      dot.style.transform = `translate(${x}px, ${y}px)`;
    }
    let segs;
    if (ed.side) {
      segs = [["nose", "head"], ["head", "withers"], ["withers", "rump"], ["rump", "tail_tip"],
        ["withers", "front_shoulder"], ["front_shoulder", "front_elbow"], ["front_elbow", "front_paw"],
        ["rump", "back_hip"], ["back_hip", "back_knee"], ["back_knee", "back_hock"]];
    } else {
      const mid = [(m.shoulder_l[0] + m.shoulder_r[0]) / 2, (m.shoulder_l[1] + m.shoulder_r[1]) / 2];
      segs = [["chin", mid], [mid, "groin"], ["shoulder_l", "shoulder_r"],
        ["shoulder_l", "elbow_l"], ["elbow_l", "wrist_l"], ["shoulder_r", "elbow_r"], ["elbow_r", "wrist_r"],
        ["groin", "knee_l"], ["knee_l", "ankle_l"], ["groin", "knee_r"], ["knee_r", "ankle_r"]];
    }
    segs = segs.filter(([a, b]) => (typeof a !== "string" || m[a]) && (typeof b !== "string" || m[b]));
    const xy = (p) => (typeof p === "string" ? pt(p) : this._toScreen(p[0], p[1]));
    ed.layer.querySelector("svg").innerHTML = segs
      .map(([a, b]) => { const [x1, y1] = xy(a), [x2, y2] = xy(b); return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`; })
      .join("");
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
