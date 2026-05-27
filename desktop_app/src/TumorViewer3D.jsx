import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Eye, EyeOff, Maximize2, Minimize2, RefreshCw, RotateCcw, Focus, Scissors } from "lucide-react";
import { Tip } from "./Tooltip";

const API_BASE = window.desktopRuntime?.apiBase || "http://127.0.0.1:8000";

const SHELL_KEYS = new Set(["brain","lung","liver"]);
const isShell = k => SHELL_KEYS.has(k);

const BRATS_COLORS = {
  ncr:        { hex:"#c0392b", three: new THREE.Color("#c0392b") }, 
  net:        { hex:"#a93226", three: new THREE.Color("#a93226") }, 
  et:         { hex:"#e8a000", three: new THREE.Color("#e8a000") }, 
  ed:         { hex:"#1a8fa0", three: new THREE.Color("#1a8fa0") }, 
  tc:         { hex:"#d44000", three: new THREE.Color("#d44000") }, 
  wt:         { hex:"#8040c0", three: new THREE.Color("#8040c0") }, 
  liver_tumor:{ hex:"#f87171", three: new THREE.Color("#f87171") }, 
  tumor:      { hex:"#f87171", three: new THREE.Color("#f87171") }, 
  lesion:     { hex:"#fb923c", three: new THREE.Color("#fb923c") }, 
  lung_tumor: { hex:"#fb923c", three: new THREE.Color("#fb923c") }, 
  nodule:     { hex:"#fbbf24", three: new THREE.Color("#fbbf24") }, 
};

function brats(label, apiFallback) {
  const l = (label||"").toLowerCase();
  for (const [key, val] of Object.entries(BRATS_COLORS)) {
    if (l === key || l.includes(key)) return val.three.clone();
  }
  return new THREE.Color(apiFallback || "#f87171");
}

const ORGAN_CAMERA = {
  brain: { z:250, gridY:-95, fov:38, scale:150, rootRotX: -Math.PI/2 },
  liver: { z:320, gridY:-75, fov:45, scale:130, rootRotX: 0 },
  lung:  { z:300, gridY:-85, fov:42, scale:140, rootRotX: 0 },
};


const SHELL_STYLES = {
  brain:   { color:0xbfa0b0, emissive:0x200810, roughness:0.70, metalness:0.00 },
  lung:    { color:0x6898c8, emissive:0x040e18, roughness:0.58, metalness:0.05 },
  liver:   { color:0x7a2e1c, emissive:0x300808, roughness:0.55, metalness:0.06 },
  default: { color:0x88c8e8, emissive:0x000818, roughness:0.65, metalness:0.00 },
};

function computeBounds(dataMap) {
  let mn=[Infinity,Infinity,Infinity], mx=[-Infinity,-Infinity,-Infinity];
  for (const d of Object.values(dataMap||{})) {
    const v=d.vertices||[];
    for(let i=0;i<v.length;i+=3){
      mn[0]=Math.min(mn[0],v[i]);   mx[0]=Math.max(mx[0],v[i]);
      mn[1]=Math.min(mn[1],v[i+1]); mx[1]=Math.max(mx[1],v[i+1]);
      mn[2]=Math.min(mn[2],v[i+2]); mx[2]=Math.max(mx[2],v[i+2]);
    }
  }
  if(!isFinite(mn[0])) return {center:new THREE.Vector3(),span:1};
  return {
    center:new THREE.Vector3((mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2),
    span:Math.max(mx[0]-mn[0],mx[1]-mn[1],mx[2]-mn[2])||1
  };
}
const countFaces = d => Math.round((d?.faces?.length||0)/3);
function makeClipPlane(span,pct){
  const z=THREE.MathUtils.lerp(-span*0.6,span*0.6,pct/100);
  return new THREE.Plane(new THREE.Vector3(0,0,-1),z);
}

export default function TumorViewer3D({caseId,meshApiPath,shellLabel="Shell"}) {
  const mountRef=useRef(null);
  const stageRef=useRef(null);
  const threeRef=useRef(null);

  const[meshData,      setMeshData]      =useState(null);
  const[loading,       setLoading]       =useState(true);
  const[loadMs,        setLoadMs]        =useState(0);
  const[errMsg,        setErrMsg]        =useState("");
  const[visible,       setVisible]       =useState({});
  const[autoRotate,    setAutoRotate]    =useState(true);
  const[selectedLabel, setSelectedLabel] =useState(null);
  const[clipEnabled,   setClipEnabled]   =useState(false);
  const[clipPercent,   setClipPercent]   =useState(60);
  const[showShell,     setShowShell]     =useState(true);
  const[shellOpacity,  setShellOpacity]  =useState(16);
  const[tumourOpacity, setTumourOpacity] =useState(90);
  const[wireframe,     setWireframe]     =useState(false);
  const[isFullscreen,  setIsFullscreen]  =useState(false);

  const toggleFullscreen=useCallback(()=>{
    if(!document.fullscreenElement) stageRef.current?.requestFullscreen?.();
    else document.exitFullscreen?.();
  },[]);
  useEffect(()=>{
    const cb=()=>setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange",cb);
    return()=>document.removeEventListener("fullscreenchange",cb);
  },[]);

  const summary=useMemo(()=>{
    if(!meshData)return{labels:0,faces:0};
    return{
      labels:Object.keys(meshData).filter(k=>!isShell(k)).length,
      faces:Object.values(meshData).reduce((a,d)=>a+countFaces(d),0)
    };
  },[meshData]);

  const resolvedPath=meshApiPath||`/viewer/${caseId}/mesh`;

  async function fetchMesh(){
    setLoading(true);setErrMsg("");setMeshData(null);
    const t0=performance.now();
    try{
      const res=await fetch(`${API_BASE}${resolvedPath}`);
      if(!res.ok) throw new Error(await res.text());
      const data=await res.json();
      setLoadMs(Math.round(performance.now()-t0));
      setMeshData(data);
      setVisible(prev=>{
        const n={...prev};
        for(const k of Object.keys(data||{})) if(!(k in n)) n[k]=true;
        return n;
      });
      const keys=Object.keys(data).filter(k=>!isShell(k));
      if(keys.length) setSelectedLabel(keys[0]);
    }catch(e){setErrMsg(e?.message||String(e));}
    finally{setLoading(false);}
  }
  useEffect(()=>{if(caseId)fetchMesh();},[caseId,resolvedPath]);

  useEffect(()=>{
    const container=mountRef.current;
    if(!container||!meshData) return;

    const W=Math.max(container.clientWidth||0,320);
    const H=Math.max(container.clientHeight||0,320);

    const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:"high-performance"});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    renderer.setSize(W,H);
    renderer.outputColorSpace=THREE.SRGBColorSpace;
    renderer.toneMapping=THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure=1.08;
    renderer.setClearColor(0x030709,1);
    renderer.localClippingEnabled=true;
    container.appendChild(renderer.domElement);

    const scene=new THREE.Scene();
    scene.fog=new THREE.FogExp2(0x030709, 0.0014);

    const orgCfg=ORGAN_CAMERA[shellLabel?.toLowerCase()]||ORGAN_CAMERA.brain;
    const camera=new THREE.PerspectiveCamera(orgCfg.fov,W/H,0.1,3000);
    camera.position.set(0,15,orgCfg.z);

    scene.add(new THREE.HemisphereLight(0xc0d0e8,0x180808,0.30));
    const key=new THREE.DirectionalLight(0xfff6e8,1.20); key.position.set(2,2.5,2); scene.add(key);
    const fill=new THREE.DirectionalLight(0x7ab0e0,0.35); fill.position.set(-2,-0.3,1); scene.add(fill);
    const rim=new THREE.DirectionalLight(0xffe080,0.45); rim.position.set(0,2,-2.5); scene.add(rim);
    const bounce=new THREE.DirectionalLight(0x203858,0.20); bounce.position.set(0,-2,0.5); scene.add(bounce);
    const innerGlow=new THREE.PointLight(0xff7040,0.28,130);
    scene.add(innerGlow);

    const root=new THREE.Group();
    if(orgCfg.rootRotX) root.rotation.x=orgCfg.rootRotX;
    scene.add(root);
    const{center,span}=computeBounds(meshData);
    const scale=orgCfg.scale/span;
    const clipPlane=makeClipPlane(150,clipPercent);

    const meshObjects={}, auxObjects={}, labelCenters={};
    let tumCentroid=new THREE.Vector3(), tumCount=0;

    for(const[label,data] of Object.entries(meshData)){
      const positions=new Float32Array(data.vertices.length);
      let cx=0,cy=0,cz=0;
      const n=Math.max(data.vertices.length/3,1);
      for(let i=0;i<data.vertices.length;i+=3){
        const x=(data.vertices[i]  -center.x)*scale;
        const y=(data.vertices[i+1]-center.y)*scale;
        const z=(data.vertices[i+2]-center.z)*scale;
        positions[i]=x; positions[i+1]=y; positions[i+2]=z;
        cx+=x; cy+=y; cz+=z;
      }
      const lc=new THREE.Vector3(cx/n,cy/n,cz/n);
      labelCenters[label]=lc;
      if(!isShell(label)){tumCentroid.add(lc);tumCount++;}

      const geo=new THREE.BufferGeometry();
      geo.setAttribute("position",new THREE.BufferAttribute(positions,3));
      geo.setIndex(data.faces);
      geo.computeVertexNormals();

      const shellKey=isShell(label)?label:null;
      const st=shellKey?(SHELL_STYLES[shellKey]||SHELL_STYLES.default):null;

      let mat;
      if(st){
        const isLiver = label === "liver";
        mat=new THREE.MeshPhysicalMaterial({
          color:     st.color,
          emissive:  st.emissive,
          roughness: st.roughness,
          metalness: st.metalness,
          transparent:true,
          opacity:   shellOpacity/100,
          side:      THREE.FrontSide,
          depthWrite:false,
          clearcoat:          isLiver ? 0.60 : 0.30,
          clearcoatRoughness: isLiver ? 0.25 : 0.45,
          clippingPlanes:clipEnabled?[clipPlane]:[],
        });
      }else{
        const baseCol=brats(label,data.color);
        const emisCol=baseCol.clone().multiplyScalar(0.18);
        mat=new THREE.MeshStandardMaterial({
          color:     baseCol,
          emissive:  emisCol,
          roughness: 0.38,
          metalness: 0.04,
          transparent:true,
          opacity:   tumourOpacity/100,
          side:      THREE.DoubleSide,
          depthWrite:true,
          clippingPlanes:clipEnabled?[clipPlane]:[],
        });
      }

      const mesh=new THREE.Mesh(geo,mat);
      mesh.visible=visible[label]!==false&&(!shellKey||showShell);
      root.add(mesh);
      meshObjects[label]=mesh;

      if(st){
        const outMat=new THREE.MeshBasicMaterial({
          color: st.color,
          side:  THREE.BackSide,
          transparent:true,
          opacity:(shellOpacity/100)*0.30,
          depthWrite:false,
          clippingPlanes:clipEnabled?[clipPlane]:[],
        });
        const outline=new THREE.Mesh(geo,outMat);
        outline.scale.setScalar(1.012);
        outline.visible=mesh.visible;
        root.add(outline);
        auxObjects[label]={outline,outMat,shellStyle:st};
      }else if(wireframe){
        const edgeGeo=new THREE.EdgesGeometry(geo,20);
        const edgeMat=new THREE.LineBasicMaterial({
          color:0xffffff,transparent:true,opacity:0.25,
          clippingPlanes:clipEnabled?[clipPlane]:[],
        });
        const edges=new THREE.LineSegments(edgeGeo,edgeMat);
        edges.visible=visible[label]!==false;
        root.add(edges);
        auxObjects[label]={edges,edgeMat};
      }

    }

    if(tumCount>0){ tumCentroid.divideScalar(tumCount); innerGlow.position.copy(tumCentroid); }


    const grid=new THREE.GridHelper(240,16,0x1a2a3c,0x0c1820);
    grid.position.y=orgCfg.gridY;
    grid.material.transparent=true; grid.material.opacity=0.18;
    root.add(grid);

    const controls=new OrbitControls(camera,renderer.domElement);
    controls.enableDamping=true; controls.dampingFactor=0.06;
    controls.autoRotate=autoRotate; controls.autoRotateSpeed=0.50;
    controls.target.set(0,0,0); controls.minDistance=50; controls.maxDistance=550;
    controls.update();

    let raf=0,dirty=true,running=false;
    const requestRender=()=>{dirty=true;if(!running){running=true;raf=requestAnimationFrame(tick);}};
    const tick=()=>{
      if(!running)return;
      raf=requestAnimationFrame(tick);
      controls.update();
      const t=performance.now()*0.001;
      innerGlow.intensity=0.20+Math.sin(t*1.6)*0.10;
      if(autoRotate||dirty){renderer.render(scene,camera);dirty=false;}
      if(!autoRotate&&!dirty){running=false;cancelAnimationFrame(raf);}
    };

    const focusLabel=lbl=>{
      const p=labelCenters[lbl]; if(!p) return;
      controls.target.copy(p);
      camera.position.set(p.x+60,p.y+20,p.z+100);
      controls.update(); requestRender();
    };
    const resetView=()=>{
      controls.target.set(0,0,0);
      camera.position.set(0,15,orgCfg.z); controls.update(); requestRender();
    };

    const resize=()=>{
      const w=Math.max(container.clientWidth||0,320), h=Math.max(container.clientHeight||0,320);
      camera.aspect=w/h; camera.updateProjectionMatrix();
      renderer.setSize(w,h); requestRender();
    };
    const ro=new ResizeObserver(resize);
    ro.observe(container);
    controls.addEventListener("change",requestRender);
    requestRender();

    threeRef.current={camera,controls,meshObjects,auxObjects,focusLabel,resetView,clipPlane,requestRender};

    return()=>{
      running=false; cancelAnimationFrame(raf);
      ro.disconnect(); controls.removeEventListener("change",requestRender);
      controls.dispose(); renderer.dispose();
      for(const m of Object.values(meshObjects)){m.geometry.dispose();m.material.dispose();}
      for(const o of Object.values(auxObjects)){
        o.outline?.geometry.dispose(); o.outMat?.dispose();
        o.edges?.geometry.dispose(); o.edgeMat?.dispose();
      }
      if(renderer.domElement.parentNode===container) container.removeChild(renderer.domElement);
      threeRef.current=null;
    };
  },[meshData,autoRotate,visible,clipEnabled,clipPercent,showShell,shellOpacity,tumourOpacity,wireframe,selectedLabel,shellLabel]);

  useEffect(()=>{
    const t=threeRef.current; if(!t) return;
    for(const[lbl,m] of Object.entries(t.meshObjects)){
      if(isShell(lbl)){
        m.material.opacity=shellOpacity/100;
        const aux=t.auxObjects[lbl];
        if(aux?.outMat) aux.outMat.opacity=(shellOpacity/100)*0.30;
        if(aux?.outline) aux.outline.visible=m.visible;
      }else{
        m.material.opacity=tumourOpacity/100;
        const aux=t.auxObjects[lbl];
      }
      m.visible=visible[lbl]!==false&&(!isShell(lbl)||showShell);
      const aux=t.auxObjects[lbl];
      if(aux?.edges) aux.edges.visible=m.visible;
    }
    t.requestRender?.();
  },[shellOpacity,tumourOpacity,visible,showShell]);

  useEffect(()=>{
    const t=threeRef.current;
    if(!t||!selectedLabel||visible[selectedLabel]===false) return;
    t.focusLabel(selectedLabel);
  },[selectedLabel]);

  function toggleLabel(lbl){setVisible(p=>({...p,[lbl]:!p[lbl]}));}

  if(loading) return(
    <div className="pane-center">
      <RefreshCw size={22} className="spin" style={{color:"var(--cyan)"}}/>
      <p style={{color:"var(--text-2)"}}>Building 3D mesh…</p>
    </div>
  );
  if(errMsg||!meshData) return(
    <div className="pane-center">
      <p style={{color:"var(--danger)",fontSize:13,maxWidth:420}}>{errMsg||"Mesh unavailable."}</p>
      <button className="btn-refresh" onClick={fetchMesh} style={{marginTop:8}}>
        <RefreshCw size={12}/> Retry
      </button>
    </div>
  );

  return(
    <div className="viewer3d-shell">
      <div className="viewer-toolbar viewer-toolbar--3d">
        <div className="viewer-seq-group viewer-seq-group--3d">
          {Object.entries(meshData).map(([lbl,data])=>{
            const on=visible[lbl]!==false;
            const dispColor=isShell(lbl)?data.color:brats(lbl,data.color).getStyle();
            return(
              <button key={lbl} type="button"
                className={`seq-btn${on?" seq-btn--active":""}`}
                style={on?{borderColor:dispColor,color:dispColor,background:`${dispColor}22`}:{}}
                onClick={()=>{toggleLabel(lbl);setSelectedLabel(lbl);}}
                title={data.name}>
                {on?<Eye size={11}/>:<EyeOff size={11}/>} {data.name}
              </button>
            );
          })}
        </div>
        <button type="button" className={`overlay-btn${autoRotate?" overlay-btn--on":""}`}
          onClick={()=>setAutoRotate(v=>!v)}>Auto-rotate {autoRotate?"ON":"OFF"}</button>
        <button type="button" className="overlay-btn"
          onClick={()=>selectedLabel&&threeRef.current?.focusLabel?.(selectedLabel)}>
          <Focus size={12}/> Focus
        </button>
        <button type="button" className={`overlay-btn${clipEnabled?" overlay-btn--on":""}`}
          onClick={()=>setClipEnabled(v=>!v)}>
          <Scissors size={12}/> Clip {clipEnabled?"ON":"OFF"}
        </button>
        <button type="button" className={`overlay-btn${showShell?" overlay-btn--on":""}`}
          onClick={()=>setShowShell(v=>!v)}>
          {shellLabel} {showShell?"ON":"OFF"}
        </button>
        <button type="button" className={`overlay-btn${wireframe?" overlay-btn--on":""}`}
          onClick={()=>setWireframe(v=>!v)}>Edges {wireframe?"ON":"OFF"}</button>
        <Tip content="Reset camera" side="bottom">
          <button type="button" className="overlay-btn" onClick={()=>threeRef.current?.resetView?.()}>
            <RotateCcw size={12}/>
          </button>
        </Tip>
        <Tip content={isFullscreen?"Exit fullscreen":"Fullscreen"} side="bottom">
          <button type="button" className={`overlay-btn${isFullscreen?" overlay-btn--on":""}`} onClick={toggleFullscreen}>
            {isFullscreen?<Minimize2 size={12}/>:<Maximize2 size={12}/>}
          </button>
        </Tip>
        <div className="viewer-legend viewer-legend--3d">
          <span>{summary.labels} region{summary.labels!==1?"s":""}</span>
          <span>{summary.faces.toLocaleString()} faces</span>
        </div>
      </div>

      {clipEnabled&&(
        <div className="viewer3d-clipbar">
          <span className="viewer3d-clipbar__label">Clip</span>
          <input type="range" min={0} max={100} value={clipPercent} className="viewer3d-clipbar__slider"
            onChange={e=>setClipPercent(Number(e.target.value))}/>
          <span className="viewer3d-clipbar__value">{clipPercent}%</span>
        </div>
      )}
      <div className="viewer3d-clipbar">
        <span className="viewer3d-clipbar__label">Tumour Opacity</span>
        <input type="range" min={20} max={100} value={tumourOpacity} className="viewer3d-clipbar__slider"
          onChange={e=>setTumourOpacity(Number(e.target.value))}/>
        <span className="viewer3d-clipbar__value">{tumourOpacity}%</span>
      </div>
      <div className="viewer3d-clipbar">
        <span className="viewer3d-clipbar__label">Shell Opacity</span>
        <input type="range" min={0} max={40} value={shellOpacity} className="viewer3d-clipbar__slider"
          onChange={e=>setShellOpacity(Number(e.target.value))}/>
        <span className="viewer3d-clipbar__value">{shellOpacity}%</span>
      </div>

      <div ref={stageRef} className="viewer3d-stage">
        <div className="viewer3d-hud">
          <div className="viewer3d-card">
            <span className="viewer3d-card__label">Selected</span>
            <strong>{selectedLabel?(meshData[selectedLabel]?.name||selectedLabel):"-"}</strong>
          </div>
          <div className="viewer3d-card">
            <span className="viewer3d-card__label">Render</span>
            <strong>PBR + Glow</strong>
          </div>
        </div>
        <div ref={mountRef} className="viewer3d-canvas"/>
      </div>
    </div>
  );
}
