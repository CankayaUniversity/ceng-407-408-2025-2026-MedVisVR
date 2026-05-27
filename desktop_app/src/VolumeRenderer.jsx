import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "vtk.js/Sources/Rendering/Profiles/Volume";
import vtkFullScreenRenderWindow from "vtk.js/Sources/Rendering/Misc/FullScreenRenderWindow";
import vtkImageData from "vtk.js/Sources/Common/DataModel/ImageData";
import vtkDataArray from "vtk.js/Sources/Common/Core/DataArray";
import vtkVolume from "vtk.js/Sources/Rendering/Core/Volume";
import vtkVolumeMapper from "vtk.js/Sources/Rendering/Core/VolumeMapper";
import vtkColorTransferFunction from "vtk.js/Sources/Rendering/Core/ColorTransferFunction";
import vtkPiecewiseFunction from "vtk.js/Sources/Common/DataModel/PiecewiseFunction";
import vtkPlane from "vtk.js/Sources/Common/DataModel/Plane";
import { BlendMode } from "vtk.js/Sources/Rendering/Core/VolumeMapper/Constants";
import { Box, Eye, Layers, Maximize2, Minimize2, RefreshCw, RotateCcw, Scissors } from "lucide-react";
import { Tip } from "./Tooltip";

const API_BASE = window.desktopRuntime?.apiBase || "http://127.0.0.1:8000";

const PRESETS = {
  brain: {
    default: "slicer-mr-default",
    items: [
      {
        key: "slicer-mr-default",
        label: "Slicer MR-Default",
        color: [[0, 0, 0, 0], [20, 0.168627, 0, 0], [40, 0.403922, 0.145098, 0.0784314], [120, 0.780392, 0.607843, 0.380392], [220, 0.847059, 0.835294, 0.788235], [1024, 1, 1, 1]],
        opacity: [[0, 0], [20, 0], [40, 0.15], [120, 0.3], [220, 0.375], [1024, 0.5]],
        gradient: [[0, 1], [255, 1]],
        unit: 2.4,
        material: { shade: true, ambient: 0.2, diffuse: 1, specular: 0, specularPower: 1 },
      },
      {
        key: "brain-mri",
        label: "Brain MRI",
        color: [[0, 0, 0, 0], [32, 0.12, 0.16, 0.2], [92, 0.45, 0.58, 0.68], [180, 0.86, 0.92, 0.98], [255, 1, 1, 1]],
        opacity: [[0, 0], [18, 0], [46, 0.025], [116, 0.13], [255, 0.46]],
        unit: 2.1,
      },
      {
        key: "brain-tumor",
        label: "Tumor Focus",
        color: [[0, 0, 0, 0], [44, 0.08, 0.12, 0.18], [110, 0.42, 0.65, 0.84], [190, 0.94, 0.86, 0.62], [255, 1, 1, 1]],
        opacity: [[0, 0], [36, 0], [86, 0.075], [156, 0.25], [255, 0.66]],
        unit: 1.5,
      },
    ],
  },
  lung: {
    default: "slicer-ct-cardiac3",
    items: [
      {
        key: "slicer-ct-air",
        label: "Slicer CT-Air",
        color: [[-3024, 1, 1, 1], [-900, 0.2, 1, 1], [-500, 0.3, 0.3, 1], [3071, 0, 0, 0]],
        opacity: [[-3024, 0.705882], [-900, 0.715686], [-500, 0], [3071, 0]],
        gradient: [[0, 1], [255, 1]],
        unit: 2.8,
        material: { shade: true, ambient: 0.1, diffuse: 0.9, specular: 0.2, specularPower: 10 },
      },
      {
        key: "slicer-ct-cardiac3",
        label: "Slicer CT-Cardiac3",
        color: [[-3024, 0, 0, 0], [-86.9767, 0, 0.25098, 1], [45.3791, 1, 0, 0], [139.919, 1, 0.894893, 0.894893], [347.907, 1, 1, 0.25098], [1224.16, 1, 1, 1], [3071, 0.827451, 0.658824, 1]],
        opacity: [[-3024, 0], [-86.9767, 0], [45.3791, 0.169643], [139.919, 0.589286], [347.907, 0.607143], [1224.16, 0.607143], [3071, 0.616071]],
        gradient: [[0, 1], [255, 1]],
        unit: 2.4,
        material: { shade: true, ambient: 0.1, diffuse: 0.9, specular: 0.2, specularPower: 10 },
      },
      {
        key: "lung-ct",
        label: "Lung CT",
        color: [[-1024, 0, 0, 0], [-920, 0.08, 0.11, 0.13], [-650, 0.42, 0.62, 0.72], [-320, 0.82, 0.9, 0.92], [180, 1, 0.96, 0.88], [1200, 1, 1, 1]],
        opacity: [[-1024, 0], [-950, 0], [-780, 0.018], [-520, 0.055], [-320, 0.09], [120, 0.012], [700, 0.075], [1600, 0.24]],
        unit: 5.6,
      },
      {
        key: "bone-ct",
        label: "Bone CT",
        color: [[-1024, 0, 0, 0], [120, 0.38, 0.28, 0.22], [450, 0.82, 0.72, 0.58], [1200, 1, 0.95, 0.82], [2400, 1, 1, 1]],
        opacity: [[-1024, 0], [120, 0], [360, 0.035], [820, 0.18], [1800, 0.48]],
        unit: 3.2,
      },
      {
        key: "soft-ct",
        label: "Soft Tissue",
        color: [[-180, 0.08, 0.06, 0.06], [-40, 0.52, 0.3, 0.24], [70, 0.86, 0.55, 0.42], [180, 1, 0.82, 0.64], [500, 1, 0.96, 0.88]],
        opacity: [[-180, 0], [-40, 0.012], [45, 0.075], [150, 0.13], [500, 0.09]],
        unit: 4.0,
      },
    ],
  },
  liver: {
    default: "slicer-ct-cardiac3",
    items: [
      {
        key: "slicer-ct-cardiac3",
        label: "Slicer CT-Cardiac3",
        color: [[-3024, 0, 0, 0], [-86.9767, 0, 0.25098, 1], [45.3791, 1, 0, 0], [139.919, 1, 0.894893, 0.894893], [347.907, 1, 1, 0.25098], [1224.16, 1, 1, 1], [3071, 0.827451, 0.658824, 1]],
        opacity: [[-3024, 0], [-86.9767, 0], [45.3791, 0.169643], [139.919, 0.589286], [347.907, 0.607143], [1224.16, 0.607143], [3071, 0.616071]],
        gradient: [[0, 1], [255, 1]],
        unit: 2.4,
        material: { shade: true, ambient: 0.1, diffuse: 0.9, specular: 0.2, specularPower: 10 },
      },
      {
        key: "liver-ct",
        label: "Liver CT",
        color: [[-200, 0, 0, 0], [-40, 0.18, 0.1, 0.08], [45, 0.7, 0.38, 0.28], [115, 0.96, 0.62, 0.42], [320, 1, 0.92, 0.72], [900, 1, 1, 1]],
        opacity: [[-200, 0], [-20, 0.01], [40, 0.07], [105, 0.14], [260, 0.11], [900, 0.22]],
        unit: 3.8,
      },
      {
        key: "soft-ct",
        label: "Soft Tissue",
        color: [[-180, 0.08, 0.06, 0.06], [-40, 0.52, 0.3, 0.24], [70, 0.86, 0.55, 0.42], [180, 1, 0.82, 0.64], [500, 1, 0.96, 0.88]],
        opacity: [[-180, 0], [-40, 0.012], [45, 0.075], [150, 0.13], [500, 0.09]],
        unit: 4.0,
      },
    ],
  },
};

const LABEL_COLORS = {
  85:  [0.74, 0.34, 0.38],  
  128: [0.82, 0.48, 0.22],  
  170: [0.78, 0.58, 0.34],  
  255: [0.54, 0.62, 0.86],  
};

function decodeBase64Bytes(b64) {
  const binary = atob(b64 || "");
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function makeTypedArray(b64, type) {
  const bytes = decodeBase64Bytes(b64);
  if (type === "int16") return new Int16Array(bytes.buffer);
  return bytes;
}

function makeImageData({ dims, spacing, values, name }) {
  const image = vtkImageData.newInstance();
  image.setDimensions(dims);
  image.setSpacing(spacing || [1, 1, 1]);
  image.getPointData().setScalars(vtkDataArray.newInstance({
    name,
    numberOfComponents: 1,
    values,
  }));
  return image;
}

function buildTransferFunctions(preset, opacityScale) {
  const color = vtkColorTransferFunction.newInstance();
  preset.color.forEach(([x, r, g, b]) => color.addRGBPoint(x, r, g, b));

  const opacity = vtkPiecewiseFunction.newInstance();
  preset.opacity.forEach(([x, a]) => opacity.addPoint(x, Math.min(1, a * opacityScale)));
  return { color, opacity };
}

function buildLabelFunctions(legend = [], maskOpacity, modality = "") {
  const color = vtkColorTransferFunction.newInstance();
  const opacity = vtkPiecewiseFunction.newInstance();
  color.addRGBPoint(0, 0, 0, 0);
  opacity.addPoint(0, 0);

  const isBrain = modality === "brain";

  const items = [...legend].sort((a, b) => Number(a.value || 0) - Number(b.value || 0));
  items.forEach((item, index) => {
    const value = Number(item.value || 0);
    const prev = Number(items[index - 1]?.value || 0);
    const low = Math.max(1, Math.round(prev + (value - prev) * 0.38));
    const shoulder = Math.max(low + 1, Math.round(prev + (value - prev) * 0.72));
    const name = String(item.name || "").toLowerCase();

    const isLiverParenchyma = name === "liver";
    const isBrainLabel      = isBrain && (name === "ncr" || name === "ed" || name === "et");
    const opacityScale = isLiverParenchyma ? 0.18 : isBrainLabel ? 0.65 : 1.0;

    const rgb = name === "ncr"         ? [0.64, 0.28, 0.32]   
      : name === "ed"                  ? [0.64, 0.50, 0.30]   
      : name === "et"                  ? [0.44, 0.56, 0.78]   
      : name === "liver"               ? [0.82, 0.48, 0.22]   
      : name === "liver tumor"         ? [0.92, 0.36, 0.36]   
      : name === "lung tumor"          ? [0.96, 0.46, 0.14]   
      : (LABEL_COLORS[value]          || [0.95, 0.44, 0.22]);

    color.addRGBPoint(Math.max(0, low - 1), 0, 0, 0);
    color.addRGBPoint(shoulder, rgb[0], rgb[1], rgb[2]);
    color.addRGBPoint(value, rgb[0], rgb[1], rgb[2]);
    color.addRGBPoint(Math.min(255, value + 1), rgb[0], rgb[1], rgb[2]);
    opacity.addPoint(Math.max(0, low - 1), 0);
    opacity.addPoint(shoulder, maskOpacity * 0.42 * opacityScale);
    opacity.addPoint(value, maskOpacity * opacityScale);
    opacity.addPoint(Math.min(255, value + 1), maskOpacity * opacityScale);
  });
  return { color, opacity };
}

function getVolumeUrl({ caseId, modality, sequence, maxDim }) {
  if (modality === "brain") {
    const seq = sequence || "t1ce";
    return `${API_BASE}/viewer/${caseId}/volume?sequence=${seq}&max_dim=${maxDim}`;
  }
  return `${API_BASE}/viewer/${modality}/${caseId}/volume?max_dim=${maxDim}`;
}

export default function VolumeRenderer({ caseId, meta, sequence, modality }) {
  const mountRef = useRef(null);
  const stageRef = useRef(null);
  const vtkRef = useRef(null);
  const autoRotateRef = useRef(true);

  const presetGroup = PRESETS[modality] || PRESETS.brain;
  const [presetKey, setPresetKey] = useState(presetGroup.default);
  const [opacity, setOpacity] = useState(modality === "brain" ? 70 : 42);
  const [maskOpacity, setMaskOpacity] = useState(modality === "brain" ? 24 : 32);
  const [roi, setRoi] = useState({ xMin: 0, xMax: 100, yMin: 0, yMax: 100, zMin: 0, zMax: 100 });
  const [showOverlay, setShowOverlay] = useState(true);
  const [autoRotate, setAutoRotate] = useState(true);
  const [maxDim, setMaxDim] = useState(modality === "brain" ? 160 : 224);
  const [loading, setLoading] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const [volumeInfo, setVolumeInfo] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      stageRef.current?.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  }, []);

  useEffect(() => {
    const onChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  const presets = presetGroup.items;
  const maxDimLimit = modality === "brain" ? 160 : 224;
  const activePreset = useMemo(
    () => presets.find((item) => item.key === presetKey) || presets[0],
    [presetKey, presets]
  );

  useEffect(() => {
    setPresetKey((PRESETS[modality] || PRESETS.brain).default);
    setOpacity(modality === "brain" ? 70 : 42);
    setMaskOpacity(modality === "brain" ? 24 : 32);
    setShowOverlay(true);
    setMaxDim(modality === "brain" ? 160 : 224);
    setRoi({ xMin: 0, xMax: 100, yMin: 0, yMax: 100, zMin: 0, zMax: 100 });
  }, [modality]);

  useEffect(() => {
    autoRotateRef.current = autoRotate;
  }, [autoRotate]);

  const disposeVtk = useCallback(() => {
    const t = vtkRef.current;
    if (!t) return;
    t.running = false;
    t.resizeObserver?.disconnect();
    t.fullScreenRenderer?.delete();
    vtkRef.current = null;
  }, []);

  const applyPreset = useCallback(() => {
    const t = vtkRef.current;
    if (!t?.volumeActor) return;
    const scale = opacity / 100;
    const { color, opacity: ofun } = buildTransferFunctions(activePreset, scale);
    const prop = t.volumeActor.getProperty();
    prop.setRGBTransferFunction(0, color);
    prop.setScalarOpacity(0, ofun);
    prop.setScalarOpacityUnitDistance(0, activePreset.unit);
    prop.setInterpolationTypeToLinear();
    prop.setShade(activePreset.material?.shade ?? true);
    prop.setAmbient(activePreset.material?.ambient ?? 0.22);
    prop.setDiffuse(activePreset.material?.diffuse ?? 0.76);
    prop.setSpecular(activePreset.material?.specular ?? 0.18);
    prop.setSpecularPower(activePreset.material?.specularPower ?? 12);
    t.renderWindow.render();
  }, [activePreset, opacity]);

  const applyLabelOverlay = useCallback(() => {
    const t = vtkRef.current;
    if (!t?.labelActor) return;
    const { color, opacity: ofun } = buildLabelFunctions(t.legend, showOverlay ? maskOpacity / 100 : 0, modality);
    const prop = t.labelActor.getProperty();
    prop.setRGBTransferFunction(0, color);
    prop.setScalarOpacity(0, ofun);
    prop.setScalarOpacityUnitDistance(0, 1);
    prop.setInterpolationTypeToLinear();
    prop.setShade(false);
    t.labelActor.setVisibility(showOverlay);
    t.renderWindow.render();
  }, [maskOpacity, showOverlay]);

  const applyMode = useCallback(() => {
    const t = vtkRef.current;
    if (!t?.volumeMapper) return;
    t.volumeMapper.setBlendMode(BlendMode.COMPOSITE_BLEND);
    t.renderWindow.render();
  }, []);

  const applyRoi = useCallback(() => {
    const t = vtkRef.current;
    if (!t?.roiPlanes || !t?.dims || !t?.spacing) return;
    const ext = [
      (t.dims[0] - 1) * t.spacing[0],
      (t.dims[1] - 1) * t.spacing[1],
      (t.dims[2] - 1) * t.spacing[2],
    ];
    const xMin = Math.min(roi.xMin, roi.xMax - 1) / 100 * ext[0];
    const xMax = Math.max(roi.xMax, roi.xMin + 1) / 100 * ext[0];
    const yMin = Math.min(roi.yMin, roi.yMax - 1) / 100 * ext[1];
    const yMax = Math.max(roi.yMax, roi.yMin + 1) / 100 * ext[1];
    const zMin = Math.min(roi.zMin, roi.zMax - 1) / 100 * ext[2];
    const zMax = Math.max(roi.zMax, roi.zMin + 1) / 100 * ext[2];
    t.roiPlanes.xMin.setOrigin(xMin, 0, 0);
    t.roiPlanes.xMax.setOrigin(xMax, 0, 0);
    t.roiPlanes.yMin.setOrigin(0, yMin, 0);
    t.roiPlanes.yMax.setOrigin(0, yMax, 0);
    t.roiPlanes.zMin.setOrigin(0, 0, zMin);
    t.roiPlanes.zMax.setOrigin(0, 0, zMax);
    t.renderWindow.render();
  }, [roi]);

  const updateRoi = (key, value) => {
    setRoi((prev) => {
      const next = { ...prev, [key]: Number(value) };
      const axis = key[0];
      const minKey = `${axis}Min`;
      const maxKey = `${axis}Max`;
      if (next[maxKey] - next[minKey] < 1) {
        if (key.endsWith("Min")) next[maxKey] = Math.min(100, next[minKey] + 1);
        else next[minKey] = Math.max(0, next[maxKey] - 1);
      }
      return next;
    });
  };

  const buildScene = useCallback(async () => {
    const container = mountRef.current;
    if (!container || !caseId || !meta) return;

    disposeVtk();
    setLoading(true);
    setReady(false);
    setError("");
    setVolumeInfo(null);

    try {
      const url = getVolumeUrl({ caseId, modality, sequence, maxDim });
      const payload = await fetch(url).then((r) => {
        if (!r.ok) throw new Error(`Volume endpoint failed (${r.status})`);
        return r.json();
      });

      const dims = payload.dims || [64, 64, 64];
      const spacing = payload.spacing || [1, 1, 1];
      const scalarValues = makeTypedArray(payload.scalar_b64 || payload.mri_b64 || payload.ct_b64, payload.scalar_type);
      const labelValues = makeTypedArray(payload.label_b64 || payload.mask_b64, "uint8");
      const scalarImage = makeImageData({ dims, spacing, values: scalarValues, name: "scalars" });
      const labelImage = makeImageData({ dims, spacing, values: labelValues, name: "labels" });

      const fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
        rootContainer: container,
        containerStyle: { width: "100%", height: "100%", position: "absolute", inset: "0", overflow: "hidden" },
        background: [0.012, 0.02, 0.035],
      });
      const renderer = fullScreenRenderer.getRenderer();
      const renderWindow = fullScreenRenderer.getRenderWindow();

      const volumeMapper = vtkVolumeMapper.newInstance();
      volumeMapper.setInputData(scalarImage);
      volumeMapper.setSampleDistance(modality === "brain" ? 0.55 : 0.42);
      volumeMapper.setImageSampleDistance(0.75);
      volumeMapper.setMaximumSamplesPerRay(2200);

      const volumeActor = vtkVolume.newInstance();
      volumeActor.setMapper(volumeMapper);
      renderer.addVolume(volumeActor);

      const labelMapper = vtkVolumeMapper.newInstance();
      labelMapper.setInputData(labelImage);
      labelMapper.setSampleDistance(0.38);
      labelMapper.setImageSampleDistance(0.7);
      labelMapper.setBlendMode(BlendMode.COMPOSITE_BLEND);

      const labelActor = vtkVolume.newInstance();
      labelActor.setMapper(labelMapper);
      renderer.addVolume(labelActor);

      const roiPlanes = {
        xMin: vtkPlane.newInstance({ normal: [1, 0, 0], origin: [0, 0, 0] }),
        xMax: vtkPlane.newInstance({ normal: [-1, 0, 0], origin: [0, 0, 0] }),
        yMin: vtkPlane.newInstance({ normal: [0, 1, 0], origin: [0, 0, 0] }),
        yMax: vtkPlane.newInstance({ normal: [0, -1, 0], origin: [0, 0, 0] }),
        zMin: vtkPlane.newInstance({ normal: [0, 0, 1], origin: [0, 0, 0] }),
        zMax: vtkPlane.newInstance({ normal: [0, 0, -1], origin: [0, 0, 0] }),
      };
      Object.values(roiPlanes).forEach((plane) => {
        volumeMapper.addClippingPlane(plane);
        labelMapper.addClippingPlane(plane);
      });

      renderer.resetCamera();
      const camera = renderer.getActiveCamera();
      if (modality === "brain") {
        const fp = camera.getFocalPoint();
        const d  = camera.getDistance();
        camera.setPosition(fp[0], fp[1] + d, fp[2]);
        camera.setViewUp(0, 0, 1);
        camera.azimuth(-15);
        camera.zoom(1.12);
      } else {
        camera.elevation(12);
        camera.azimuth(-24);
        camera.zoom(1.12);
      }
      renderer.resetCameraClippingRange();

      const resizeObserver = new ResizeObserver(() => {
        fullScreenRenderer.resize();
        renderWindow.render();
      });
      resizeObserver.observe(container);

      vtkRef.current = {
        fullScreenRenderer,
        renderer,
        renderWindow,
        volumeMapper,
        volumeActor,
        labelMapper,
        labelActor,
        roiPlanes,
        dims,
        spacing,
        legend: payload.label_legend || [],
        resizeObserver,
        running: true,
      };

      setVolumeInfo({
        dims,
        spacing,
        downsample: payload.downsample || 1,
        scalarType: payload.scalar_type || "uint8",
        hasMask: Boolean(payload.has_mask),
      });
      setReady(true);
      setLoading(false);

      const { color, opacity: ofun } = buildTransferFunctions(activePreset, opacity / 100);
      volumeActor.getProperty().setRGBTransferFunction(0, color);
      volumeActor.getProperty().setScalarOpacity(0, ofun);
      volumeActor.getProperty().setScalarOpacityUnitDistance(0, activePreset.unit);
      volumeActor.getProperty().setInterpolationTypeToLinear();
      volumeActor.getProperty().setShade(activePreset.material?.shade ?? true);
      volumeActor.getProperty().setAmbient(activePreset.material?.ambient ?? 0.22);
      volumeActor.getProperty().setDiffuse(activePreset.material?.diffuse ?? 0.76);
      volumeActor.getProperty().setSpecular(activePreset.material?.specular ?? 0.18);
      volumeActor.getProperty().setSpecularPower(activePreset.material?.specularPower ?? 12);

      const { color: labelColor, opacity: labelOpacity } = buildLabelFunctions(payload.label_legend || [], showOverlay ? maskOpacity / 100 : 0, modality);
      labelActor.getProperty().setRGBTransferFunction(0, labelColor);
      labelActor.getProperty().setScalarOpacity(0, labelOpacity);
      labelActor.getProperty().setScalarOpacityUnitDistance(0, 1);
      labelActor.getProperty().setInterpolationTypeToLinear();
      labelActor.getProperty().setShade(false);
      labelActor.setVisibility(showOverlay);

      volumeMapper.setBlendMode(BlendMode.COMPOSITE_BLEND);
      const ext = [(dims[0] - 1) * spacing[0], (dims[1] - 1) * spacing[1], (dims[2] - 1) * spacing[2]];
      roiPlanes.xMin.setOrigin(roi.xMin / 100 * ext[0], 0, 0);
      roiPlanes.xMax.setOrigin(roi.xMax / 100 * ext[0], 0, 0);
      roiPlanes.yMin.setOrigin(0, roi.yMin / 100 * ext[1], 0);
      roiPlanes.yMax.setOrigin(0, roi.yMax / 100 * ext[1], 0);
      roiPlanes.zMin.setOrigin(0, 0, roi.zMin / 100 * ext[2]);
      roiPlanes.zMax.setOrigin(0, 0, roi.zMax / 100 * ext[2]);
      renderWindow.render();

      const tick = () => {
        const t = vtkRef.current;
        if (!t?.running) return;
        if (autoRotateRef.current) {
          t.renderer.getActiveCamera().azimuth(0.12);
          t.renderer.resetCameraClippingRange();
          t.renderWindow.render();
        }
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    } catch (exc) {
      disposeVtk();
      setLoading(false);
      setReady(false);
      setError(exc?.message || "Volume could not be loaded.");
    }
  }, [caseId, disposeVtk, maxDim, meta, modality, sequence]);

  useEffect(() => {
    buildScene();
    return disposeVtk;
  }, [buildScene, disposeVtk]);

  useEffect(() => { applyPreset(); }, [applyPreset]);
  useEffect(() => { applyLabelOverlay(); }, [applyLabelOverlay]);
  useEffect(() => { applyMode(); }, [applyMode]);
  useEffect(() => { applyRoi(); }, [applyRoi]);

  const resetCamera = () => {
    const t = vtkRef.current;
    if (!t) return;
    t.renderer.resetCamera();
    const cam = t.renderer.getActiveCamera();
    if (modality === "brain") {
      const fp = cam.getFocalPoint();
      const d  = cam.getDistance();
      cam.setPosition(fp[0], fp[1] + d, fp[2]);
      cam.setViewUp(0, 0, 1);
      cam.azimuth(-15);
      cam.zoom(1.12);
    } else {
      cam.elevation(12);
      cam.azimuth(-24);
      cam.zoom(1.12);
    }
    t.renderer.resetCameraClippingRange();
    t.renderWindow.render();
  };

  return (
    <div className="vol-shell">
      <div className="vol-toolbar">
        <Tip content="Auto rotate" side="bottom">
          <button type="button" className={`overlay-btn${autoRotate ? " overlay-btn--on" : ""}`} onClick={() => setAutoRotate((v) => !v)}>
            <RefreshCw size={12} />
          </button>
        </Tip>
        <Tip content="Segmentation overlay" side="bottom">
          <button type="button" className={`overlay-btn${showOverlay ? " overlay-btn--on" : ""}`} onClick={() => setShowOverlay((v) => !v)}>
            <Layers size={12} />
          </button>
        </Tip>
        <Tip content="Reset camera" side="bottom">
          <button type="button" className="overlay-btn" onClick={resetCamera}>
            <RotateCcw size={12} />
          </button>
        </Tip>
        <Tip content={isFullscreen ? "Exit fullscreen" : "Fullscreen"} side="bottom">
          <button type="button" className={`overlay-btn${isFullscreen ? " overlay-btn--on" : ""}`} onClick={toggleFullscreen}>
            {isFullscreen ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
          </button>
        </Tip>

        <select className="vol-select" value={presetKey} onChange={(e) => setPresetKey(e.target.value)} title="Transfer preset">
          {presets.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}
        </select>

        <span className="vol-mode-static">Composite</span>

        {(ready || loading) && (
          <span className="vol-dims">
            {volumeInfo?.dims?.join("x") || `${meta?.shape?.x ?? "?"}x${meta?.shape?.y ?? "?"}x${meta?.shape?.z ?? "?"}`}
          </span>
        )}
      </div>

      <div className="vol-ctrl-row">
        <div className="vol-slider-group">
          <Eye size={12} />
          <label title="Volume opacity">Vol</label>
          <input type="range" min={20} max={160} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
          <span>{opacity}%</span>
        </div>
        <div className="vol-slider-group">
          <Layers size={12} />
          <label title="Mask opacity">Mask</label>
          <input type="range" min={0} max={100} value={maskOpacity} onChange={(e) => setMaskOpacity(Number(e.target.value))} />
          <span>{maskOpacity}%</span>
        </div>
        <div className="vol-slider-group">
          <Box size={12} />
          <label title="Voxel budget">Dim</label>
          <input type="range" min={96} max={maxDimLimit} step={32} value={maxDim} onChange={(e) => setMaxDim(Number(e.target.value))} />
          <span>{maxDim}</span>
        </div>
      </div>

      <div className="vol-roi-row">
        <div className="vol-roi-title"><Scissors size={12} /> Display ROI</div>
        {[
          ["x", "X"],
          ["y", "Y"],
          ["z", "Z"],
        ].map(([axis, label]) => (
          <div className="vol-roi-axis" key={axis}>
            <span>{label}</span>
            <input
              type="range"
              min={0}
              max={99}
              value={roi[`${axis}Min`]}
              onChange={(e) => updateRoi(`${axis}Min`, e.target.value)}
              title={`${label} minimum`}
            />
            <input
              type="range"
              min={1}
              max={100}
              value={roi[`${axis}Max`]}
              onChange={(e) => updateRoi(`${axis}Max`, e.target.value)}
              title={`${label} maximum`}
            />
            <strong>{roi[`${axis}Min`]}-{roi[`${axis}Max`]}%</strong>
          </div>
        ))}
      </div>

      <div ref={stageRef} className="vol-stage">
        <div ref={mountRef} className="vol-canvas" />

        {loading && (
          <div className="vol-progress">
            <RefreshCw size={20} className="spin" style={{ color: "var(--accent)" }} />
            <div className="vol-progress__label">Preparing GPU volume</div>
            <div className="vol-progress__bar">
              <div className="vol-progress__fill" style={{ width: "72%" }} />
            </div>
          </div>
        )}

        {!loading && error && (
          <div className="pane-center">
            <p style={{ color: "var(--danger)", fontSize: 13 }}>{error}</p>
            <button className="btn-refresh" onClick={buildScene} style={{ marginTop: 8 }}>
              <RefreshCw size={12} /> Retry
            </button>
          </div>
        )}

      </div>
    </div>
  );
}
