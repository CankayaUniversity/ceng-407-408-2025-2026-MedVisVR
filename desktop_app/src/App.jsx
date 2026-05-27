import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  AlertTriangle, BadgeCheck, Brain,
  Eye, EyeOff, FileText, Layers,
  MessageSquare, RefreshCw, Search, Send,
  ShieldAlert, FolderOpen,
  Printer, Download, Sun, Moon, Upload,
  StickyNote, Highlighter, X, ChevronDown, ChevronUp,
  Activity, Zap, ChevronLeft, ChevronRight, Maximize2, Minimize2,
  Info, Filter, SortAsc, ArrowDownAZ, ArrowUpAZ, PanelRight, PanelRightClose,
  Stethoscope, FlaskConical, Scan, BarChart3,
} from "lucide-react";
import { Toaster, toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import { Tip, TooltipProvider } from "./Tooltip";

import { askQuestion, clearQASession, getCaseDetail, getCases, regenerateReport } from "./api";
import { BRAIN_PROMPTS, CT_PROMPTS } from "./samplePrompts";
import TumorViewer3D from "./TumorViewer3D";
import VolumeRenderer from "./VolumeRenderer";
import { downloadReportAsPdf, saveReportToFile } from "./pdf_export";

const API_BASE = window.desktopRuntime?.apiBase || "http://127.0.0.1:8000";

async function apiFetch(path, opts={}) {
  const res = await fetch(`${API_BASE}${path}`,{headers:{"Content-Type":"application/json"},...opts});
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

const CarvisLogoSmall = () => (
  <div style={{width:30,height:30,borderRadius:"50%",overflow:"hidden",flexShrink:0}}>
    <img src="./lumina_logo.png" alt="CARVIS" style={{width:"100%",height:"100%",display:"block",objectFit:"cover"}}/>
  </div>
);

const CarvisLogo = ({ size = 160 }) => (
  <div style={{
    width: size,
    height: size,
    borderRadius: "50%",
    overflow: "hidden",
    flexShrink: 0,
    display: "block",
  }}>
    <img
      src="./lumina_logo.png"
      alt="CARVIS"
      style={{width:"100%", height:"100%", display:"block", objectFit:"cover"}}
    />
  </div>
);

const LungIcon = ({size=13,style={}}) => (
  <svg width={size} height={size} viewBox="0 0 20 20" fill="none" stroke="currentColor"
    strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" style={style}>
    <path d="M10 2v11"/>
    <path d="M10 7C9 6 7 5.5 6 7 5 8.5 4 11 4 14c0 1.5.8 2 1.8 2C7 16 7.5 15 8 14c.5-1 1-2.5 2-3"/>
    <path d="M10 7c1-1 3-1.5 4 1 1 1.5 2 4 2 7 0 1.5-.8 2-1.8 2C13 17 12.5 16 12 15c-.5-1-1-2.5-2-3"/>
  </svg>
);
const LiverIcon = ({size=13,style={}}) => (
  <svg width={size} height={size} viewBox="0 0 20 20" fill="none" stroke="currentColor"
    strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" style={style}>
    <path d="M4 10c0-4 2.5-6 6-6 4 0 7 2.2 7 6.5 0 3-2 5.5-5 5.5H5.5C4 16 3.5 14.5 3.5 13c0-1.5.5-2.5 1.5-3.5"/>
    <circle cx="13" cy="11" r="1" fill="currentColor"/>
  </svg>
);

function flattenNodeText(node) {
  if (typeof node === "string") return node;
  if (Array.isArray(node)) return node.map(flattenNodeText).join("");
  if (node && typeof node === "object" && node.props && node.props.children)
    return flattenNodeText(node.props.children);
  return "";
}

function reportLineVariant(text = "") {
  const low = String(text).toLowerCase().trim();
  if (!low) return "";

  if (
    low.includes("midline shift") || low.includes("herniation") ||
    low.includes("mass effect") || low.includes("critical") ||
    low.includes("urgent") || low.includes("high risk") ||
    low.includes("high-risk") || low.includes("surgical risk") ||
    low.includes("compression") || low.includes("effacement") ||
    low.includes("obliteration") || low.includes("invasion") ||
    low.includes("malignant") || low.includes("concerning") ||
    low.includes("abnormal") || low.includes("significant mass") ||
    low.includes("vascular involvement") || low.includes("infiltrat")
  ) return "alert";

  if (
    low.includes("volume") || low.includes("diameter") ||
    low.includes("burden") || low.includes("mm3") || low.includes("cm3") ||
    low.includes("cm³") || low.includes("mm³") ||
    low.includes("confidence:") || low.includes("measurement") ||
    low.includes("ratio") || low.includes("composition") ||
    low.includes("component") || low.includes("lesion count") ||
    low.includes("lesion architecture") || low.includes("tumour size") ||
    low.includes("tumor size") || low.includes("segmented") ||
    low.includes("parenchyma") || low.includes("maximum") ||
    low.includes("total segment") || low.includes("enhancing") ||
    low.includes("non-enhancing") || low.includes("edema") ||
    /\d+(\.\d+)?\s*(mm|cm|ml|%)/.test(low)
  ) return "metric";

  if (
    low.includes("lobe") || low.includes("location") ||
    low.includes("region") || low.includes("hemisphere") ||
    low.includes("localiz") || low.includes("overlap") ||
    low.includes("white matter") || low.includes("cortex") ||
    low.includes("segment") || low.includes("quadrant") ||
    low.includes("hepatic") || low.includes("pulmonary") ||
    low.includes("subcortical") || low.includes("subcapsular") ||
    low.includes("dominant pattern") || low.includes("approximate regional") ||
    low.includes("couinaud") || low.includes("laterality") ||
    low.includes("anatomic") || low.includes("proximity")
  ) return "location";

  if (
    low.includes("limitation") || low.includes("uncertainty") ||
    low.includes("surrogate") || low.includes("atlas") ||
    low.includes("confidence interval") || low.includes("note:") ||
    low.includes("should be treated") || low.includes("not a direct") ||
    low.includes("clinician") || low.includes("confirmation") ||
    low.includes("secondary automated") || low.includes("indirect") ||
    low.includes("approximat") || low.includes("estimate")
  ) return "limitation";

  const colonIdx = low.indexOf(":");
  if (colonIdx > 0 && colonIdx < 50 && low.length > colonIdx + 2) {
    const afterColon = low.slice(colonIdx + 1).trim();
    if (afterColon && !/^(yes|no|unknown|n\/a|none|balanced|present|absent)$/.test(afterColon))
      return "metric";
  }

  return "";
}

const _SEV_WORDS = [
  ["critical","sev-critical"],["high risk","sev-high"],["high-risk","sev-high"],
  ["warning","sev-warn"],["caution","sev-warn"],["elevated","sev-warn"],
  ["normal","sev-normal"],["within normal","sev-normal"],["no significant","sev-normal"],
];
function renderHighlightedReportText(text = "") {
  let s = String(text);
  s = s.replace(/(\d+\.?\d*)\s*(mm³?|cm³?|cm3|mm3|%|ml)/gi,
    (_, n, u) => `<span class="report-metric-value">${n} ${u}</span>`);
  for (const [w, cls] of _SEV_WORDS) {
    const re = new RegExp(`\\b(${w})\\b`, "gi");
    s = s.replace(re, `<span class="report-severity ${cls}">$1</span>`);
  }
  return s;
}

function ReportBlock({ as: Tag = "p", children }) {
  const raw = flattenNodeText(children);
  const colonIdx = raw.indexOf(":");
  const variant = reportLineVariant(raw);
  if (!variant) return <Tag className="report-line">{children}</Tag>;
  if (colonIdx > 0 && colonIdx < 60) {
    const label = raw.slice(0, colonIdx);
    const rest = raw.slice(colonIdx + 1).trim();
    return (
      <Tag className={`report-line report-line--${variant}`}>
        <span className="report-line-label">{label}:</span>{" "}
        <span dangerouslySetInnerHTML={{ __html: renderHighlightedReportText(rest) }} />
      </Tag>
    );
  }
  return (
    <Tag className={`report-line report-line--${variant}`}
      dangerouslySetInnerHTML={{ __html: renderHighlightedReportText(raw) }} />
  );
}

const reportMarkdownComponents = {
  h1: ({ children }) => (
    <h1 className="report-heading report-heading--main">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="report-heading report-heading--section">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="report-heading report-heading--subsection">{children}</h3>
  ),
  p:  ({ children }) => <ReportBlock as="p">{children}</ReportBlock>,
  li: ({ children }) => <ReportBlock as="li">{children}</ReportBlock>,
};

const toCm3 = v => v ? `${(v/1000).toFixed(2)} cm³` : "—";
function humanSeq(f) {
  const s=String(f||"").toLowerCase();
  if(s.includes("t1ce"))  return "T1ce";
  if(s.includes("flair")) return "FLAIR";
  if(s.includes("t2"))    return "T2";
  if(s.includes("t1"))    return "T1";
  return f;
}
function answerStatus(resp) {
  const a=(resp?.answer||"").trim().toLowerCase();
  if(a.startsWith("insufficient evidence")) return "insufficient";
  if(!resp?.evidence_ids?.length)           return "unsupported";
  return "supported";
}

const MODALITY_CONFIG = {
  brain:{
    label:"Brain",
    metaPath: id=>`/viewer/${id}/meta`,
    slicePath:(id,plane,idx,seq,ov)=>`/viewer/${id}/slice?plane=${plane}&index=${idx}&sequence=${seq}&overlay=${ov}`,
    meshPath:  id=>`/viewer/${id}/mesh`,
    importMaskPath:null,
    sequences:["t1","t1ce","t2","flair"], seqLabels:{t1:"T1",t1ce:"T1ce",t2:"T2",flair:"FLAIR"},
    legend:[{color:"rgb(224,92,92)",label:"NCR"},{color:"rgb(232,162,72)",label:"ED"},{color:"rgb(77,142,240)",label:"ET"}],
    shellLabel:"Brain shell", usesOwnCaseId:true, caseIdHint:"e.g. BraTS20_Validation_001",
  },
  lung:{
    label:"Lung",
    metaPath: id=>`/viewer/lung/${id}/meta`,
    slicePath:(id,plane,idx,_seq,ov)=>`/viewer/lung/${id}/slice?plane=${plane}&index=${idx}&overlay=${ov}`,
    meshPath:  id=>`/viewer/lung/${id}/mesh`,
    importMaskPath:id=>`/viewer/lung/${id}/import-mask`,
    sequences:["ct"],seqLabels:{ct:"CT"},
    legend:[{color:"rgb(249,115,22)",label:"Lung Tumor"}],
    shellLabel:"Lung shell",usesOwnCaseId:true,caseIdHint:"e.g. lung_001",
  },
  liver:{
    label:"Liver",
    metaPath: id=>`/viewer/liver/${id}/meta`,
    slicePath:(id,plane,idx,_seq,ov)=>`/viewer/liver/${id}/slice?plane=${plane}&index=${idx}&overlay=${ov}`,
    meshPath:  id=>`/viewer/liver/${id}/mesh`,
    importMaskPath:id=>`/viewer/liver/${id}/import-mask`,
    sequences:["ct"],seqLabels:{ct:"CT"},
    legend:[{color:"rgb(52,211,153)",label:"Liver"},{color:"rgb(248,113,113)",label:"Liver Tumor"}],
    shellLabel:"Organ shell",usesOwnCaseId:true,caseIdHint:"e.g. liver_001",
  },
};

function ModalityIcon({modality,size=13}) {
  if(modality==="lung")  return <LungIcon  size={size}/>;
  if(modality==="liver") return <LiverIcon size={size}/>;
  return <Brain size={size}/>;
}

const PLANES=["axial","coronal","sagittal"];
function SlicePanel({sliceUrl,plane,index,maxIndex,onChange}) {
  const[loading,setLoading]=useState(true);
  const[error,setError]=useState(false);
  return(
    <div className="slice-panel">
      <div className="slice-panel__header">
        <span className="slice-plane-label">{plane.charAt(0).toUpperCase()+plane.slice(1)}</span>
        <span className="slice-index-badge">{index}/{maxIndex}</span>
      </div>
      <div className="slice-canvas">
        {loading&&!error&&<div className="slice-loading"><RefreshCw size={16} className="spin" style={{color:"var(--blue)"}}/></div>}
        {error&&<div className="slice-loading" style={{color:"var(--text-3)"}}>No image</div>}
        <img key={sliceUrl} src={sliceUrl} alt={`${plane} ${index}`} className="slice-img"
          style={{display:loading||error?"none":"block"}}
          onLoad={()=>{setLoading(false);setError(false);}}
          onError={()=>{setLoading(false);setError(true);}}/>
      </div>
      <input type="range" className="slice-slider" min={0} max={maxIndex} value={index}
        onChange={e=>{setLoading(true);setError(false);onChange(Number(e.target.value));}}/>
    </div>
  );
}

function ViewerPane({caseId,defaultModality,defaultCaseId}) {
  const modality  = defaultModality || "brain";
  const effectiveId = modality === "brain" ? (caseId || "") : (defaultCaseId || "");

  const cfg = MODALITY_CONFIG[modality];

  const[meta,      setMeta]      =useState(null);
  const[sequence,  setSequence]  =useState("t1ce");
  const[overlay,   setOverlay]   =useState(true);
  const[indices,   setIndices]   =useState({axial:77,coronal:120,sagittal:120});
  const[loading,   setLoading]   =useState(true);
  const[errMsg,    setErrMsg]    =useState("");
  const[viewerMode,setViewerMode]=useState("slices");
  const[importBusy,setImportBusy]=useState(false);
  const[importMsg, setImportMsg] =useState("");
  const[fullscreen,setFullscreen]=useState(false);

  const loadMeta=async()=>{
    if(!effectiveId){setMeta(null);setLoading(false);setErrMsg("");return;}
    setLoading(true);setMeta(null);setErrMsg("");
    try{const m=await apiFetch(cfg.metaPath(effectiveId));setMeta(m);setIndices(m.default_indices);setSequence((m.available_sequences||[])[0]||"ct");}
    catch(e){setErrMsg(e?.message||String(e));}
    finally{setLoading(false);}
  };
  useEffect(()=>{loadMeta();},[]);

  const makeSliceUrl=plane=>{if(!meta||!effectiveId)return"";return`${API_BASE}${cfg.slicePath(effectiveId,plane,indices[plane]||0,sequence,overlay)}`;};
  const maxIdx=meta?{axial:meta.shape.z-1,coronal:meta.shape.y-1,sagittal:meta.shape.x-1}:{axial:0,coronal:0,sagittal:0};

  const handleImportMask=async()=>{
    if(!cfg.importMaskPath||!effectiveId||importBusy)return;
    setImportBusy(true);setImportMsg("");
    try{const r=await apiFetch(cfg.importMaskPath(effectiveId),{method:"POST"});setImportMsg(r.message||"Mask imported");await loadMeta();}
    catch(e){let raw=e?.message||String(e);try{raw=JSON.parse(raw)?.detail||raw;}catch{}setImportMsg(`Error: ${raw}`);}
    finally{setImportBusy(false);}
  };

  return(
    <div className={`viewer-pane${fullscreen?" viewer-pane--fullscreen":""}`}>
      <div className="viewer-subtab-bar">
        {[
          {key:"slices", label:"Slices",   icon:<Layers size={12}/>},
          {key:"3d",     label:"3D Model", icon:<ModalityIcon modality={modality} size={12}/>},
          {key:"volume", label:"Volume",   icon:<Layers size={12}/>},
        ].map(t=>(
          <button key={t.key} type="button"
            className={`viewer-subtab${viewerMode===t.key?" viewer-subtab--active":""}`}
            onClick={()=>setViewerMode(t.key)}>
            <span style={{display:"flex",alignItems:"center"}}>{t.icon}</span> {t.label}
          </button>
        ))}
      </div>

      {meta&&!meta.has_mask&&cfg.importMaskPath&&effectiveId&&(
        <div className="import-mask-bar">
          <Upload size={12}/>
          <span>No segmentation mask for <strong>{effectiveId}</strong>.</span>
          <button className="btn-import-mask" onClick={handleImportMask} disabled={importBusy}>
            {importBusy?<><RefreshCw size={11} className="spin"/> Importing…</>:<><Upload size={11}/> Import from nnUNet</>}
          </button>
          {importMsg&&<span className={`import-mask-msg${importMsg.startsWith("Error")?" import-mask-msg--err":""}`}>{importMsg}</span>}
        </div>
      )}

      {(viewerMode==="3d"||viewerMode==="volume")&&(
        <button className="viewer-fullscreen-btn" onClick={()=>setFullscreen(v=>!v)} title={fullscreen?"Exit fullscreen":"Fullscreen"}>
          {fullscreen?<Minimize2 size={14}/>:<Maximize2 size={14}/>}
        </button>
      )}

      {viewerMode==="3d"&&effectiveId&&<TumorViewer3D caseId={effectiveId} meshApiPath={cfg.meshPath(effectiveId)} shellLabel={cfg.shellLabel}/>}
      {viewerMode==="3d"&&!effectiveId&&<div className="pane-center"><ModalityIcon modality={modality} size={26}/><p style={{color:"var(--text-3)"}}>No case selected.</p></div>}

      {viewerMode==="volume"&&effectiveId&&meta&&(
        <VolumeRenderer caseId={effectiveId} meta={meta} slicePath={cfg.slicePath} sequence={sequence} modality={modality}/>
      )}
      {viewerMode==="volume"&&effectiveId&&!meta&&(
        <div className="pane-center">
          {loading
            ? <><RefreshCw size={20} className="spin" style={{color:"var(--blue)"}}/><p>Loading…</p></>
            : <><Layers size={24} style={{color:"var(--muted)"}}/><p style={{color:"var(--danger)",fontSize:13}}>{errMsg||"Could not load volume."}</p><button className="btn-refresh" onClick={loadMeta} style={{marginTop:8}}><RefreshCw size={12}/> Retry</button></>
          }
        </div>
      )}
      {viewerMode==="volume"&&!effectiveId&&<div className="pane-center"><p style={{color:"var(--text-3)"}}>No case selected.</p></div>}

      {viewerMode==="slices"&&(
        <>
          {loading&&<div className="pane-center"><RefreshCw size={20} className="spin" style={{color:"var(--blue)"}}/><p>Loading…</p></div>}
          {!loading&&!meta&&errMsg&&(
            <div className="pane-center">
              <Layers size={24} style={{color:"var(--muted)"}}/>
              <p style={{color:"var(--danger)",fontSize:13,maxWidth:460}}>{errMsg}</p>
              <button className="btn-refresh" onClick={loadMeta} style={{marginTop:8}}><RefreshCw size={12}/> Retry</button>
            </div>
          )}
          {!loading&&meta&&(
            <>
              <div className="viewer-toolbar">
                <div className="viewer-seq-group">
                  {meta.available_sequences.map(seq=>(
                    <button key={seq} type="button" className={`seq-btn${sequence===seq?" seq-btn--active":""}`} onClick={()=>setSequence(seq)}>
                      {cfg.seqLabels[seq]||seq.toUpperCase()}
                    </button>
                  ))}
                </div>
                <button type="button" className={`overlay-btn${overlay?" overlay-btn--on":""}`}
                  onClick={()=>setOverlay(v=>!v)} disabled={!meta.has_mask}
                  title={meta.has_mask?"Toggle overlay":"Import or run segmentation first"}>
                  {overlay?<Eye size={13}/>:<EyeOff size={13}/>} Overlay {overlay?"ON":"OFF"}
                </button>
                <div className="viewer-legend">
                  {cfg.legend.map(l=>(
                    <span key={l.label}><span className="legend-dot" style={{background:l.color}}/> {l.label}</span>
                  ))}
                  {!meta.has_mask&&<span style={{color:"var(--text-3)",fontSize:10.5,fontStyle:"italic"}}>no mask</span>}
                </div>
              </div>
              <div className="slice-grid">
                <SlicePanel key={`axial-${modality}-${effectiveId}`}
                  sliceUrl={makeSliceUrl("axial")} plane="axial"
                  index={indices.axial} maxIndex={maxIdx.axial}
                  onChange={v=>setIndices(prev=>({...prev,axial:v}))}/>
                <div className="slice-col-right">
                  <SlicePanel key={`coronal-${modality}-${effectiveId}`}
                    sliceUrl={makeSliceUrl("coronal")} plane="coronal"
                    index={indices.coronal} maxIndex={maxIdx.coronal}
                    onChange={v=>setIndices(prev=>({...prev,coronal:v}))}/>
                  <SlicePanel key={`sagittal-${modality}-${effectiveId}`}
                    sliceUrl={makeSliceUrl("sagittal")} plane="sagittal"
                    index={indices.sagittal} maxIndex={maxIdx.sagittal}
                    onChange={v=>setIndices(prev=>({...prev,sagittal:v}))}/>
                </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}

function WelcomeScreen({brainCount, lungCount, liverCount}) {
  const total = brainCount + lungCount + liverCount;
  return (
    <div className="welcome-screen">
      <div className="welcome-grid"/>
      <div className="welcome-glow welcome-glow--1"/>
      <div className="welcome-glow welcome-glow--2"/>
      <div className="welcome-particles">
        {[...Array(8)].map((_,i) => <span key={i} className="welcome-particle"/>)}
      </div>

      <div className="welcome-content">

        <div className="welcome-logo">
          <CarvisLogo size={120}/>
        </div>

        <div className="welcome-divider"/>

        <p className="welcome-sub">
          Clinical Anatomical Rendering Visualization Intelligent System<br/>
          AI-powered tumour segmentation · 3D anatomical visualisation
        </p>

        <div className="welcome-stats">
          <div className="welcome-stat">
            <Brain size={20} style={{color:"var(--blue-bright)"}}/>
            <span className="welcome-stat__num">{brainCount}</span>
            <span className="welcome-stat__label">Brain MRI</span>
          </div>
          <div className="welcome-stat">
            <LungIcon size={20} style={{color:"var(--lung)"}}/>
            <span className="welcome-stat__num">{lungCount}</span>
            <span className="welcome-stat__label">Lung CT</span>
          </div>
          <div className="welcome-stat">
            <LiverIcon size={20} style={{color:"var(--liver)"}}/>
            <span className="welcome-stat__num">{liverCount}</span>
            <span className="welcome-stat__label">Liver CT</span>
          </div>
        </div>

        <div className="welcome-features">
          <div className="welcome-feature">
            <span className="welcome-feature__icon" style={{color:"var(--blue-bright)"}}>
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                <circle cx="8" cy="8" r="6.5"/><circle cx="8" cy="8" r="2.5"/>
                <line x1="8" y1="1.5" x2="8" y2="4"/><line x1="8" y1="12" x2="8" y2="14.5"/>
                <line x1="1.5" y1="8" x2="4" y2="8"/><line x1="12" y1="8" x2="14.5" y2="8"/>
              </svg>
            </span>
            AI-powered tumour segmentation &amp; reporting
          </div>
          <div className="welcome-feature">
            <span className="welcome-feature__icon" style={{color:"var(--cyan)"}}><FileText size={13}/></span>
            Structured clinical reports with evidence grounding
          </div>
          <div className="welcome-feature">
            <span className="welcome-feature__icon" style={{color:"#a78bfa"}}><Activity size={13}/></span>
            Interactive 3D imaging viewer
          </div>
          <div className="welcome-feature">
            <span className="welcome-feature__icon" style={{color:"var(--teal)"}}><Zap size={13}/></span>
            Clinical Q&amp;A grounded in segmentation evidence
          </div>
        </div>

        <div className="welcome-cta">
          <span className="welcome-cta__arrow">←</span>
          {total > 0
            ? `${total} case${total !== 1 ? "s" : ""} ready · Select one from the library`
            : "Select a case from the library to begin"}
        </div>

        <p className="welcome-disclaimer">
          Decision-support tool only · Not intended for direct clinical diagnosis
        </p>
      </div>
    </div>
  );
}

function detectBrainSeq(filename) {
  const n = (filename || "").toLowerCase();
  if (n.includes("_flair")) return "flair";
  if (n.includes("_t1ce"))  return "t1ce";
  if (n.includes("_t2"))    return "t2";
  if (n.includes("_t1"))    return "t1";
  return null;
}

const BRAIN_SEQ_DEFS = [
  { key: "flair", label: "FLAIR", required: true },
  { key: "t1",    label: "T1",    required: false },
  { key: "t1ce",  label: "T1ce",  required: false },
  { key: "t2",    label: "T2",    required: false },
];

function UploadModal({ onClose, onCaseReady }) {
  const [organ,      setOrgan]      = useState("liver");
  const [filePath,   setFilePath]   = useState("");
  const [brainFiles, setBrainFiles] = useState({});
  const [isDragOver, setIsDragOver] = useState(false);
  const [phase,      setPhase]      = useState("select");
  const [progress,   setProgress]   = useState(0);
  const [message,    setMessage]    = useState("");
  const [caseId,     setCaseId]     = useState("");
  const jobRef  = useRef(null);
  const pollRef = useRef(null);

  const rt = window.desktopRuntime;

  const choosFile = async () => {
    const p = await rt?.chooseNiftiFile?.();
    if (p) setFilePath(p);
  };

  const addBrainFilePaths = (paths) => {
    setBrainFiles(prev => {
      const updated = { ...prev };
      for (const fp of paths) {
        const name = fp.split(/[\\/]/).pop();
        const seq  = detectBrainSeq(name);
        if (seq) {
          updated[seq] = fp;
        } else if (!updated.flair) {
          updated.flair = fp;
        }
      }
      return updated;
    });
  };

  const chooseBrainFiles = async () => {
    const paths = await rt?.chooseNiftiFiles?.();
    if (paths && paths.length) addBrainFilePaths(paths);
  };

  const removeBrainFile = (seq) =>
    setBrainFiles(prev => { const u = { ...prev }; delete u[seq]; return u; });

  const handleDragOver  = (e) => { e.preventDefault(); setIsDragOver(true); };
  const handleDragLeave = (e) => { if (!e.currentTarget.contains(e.relatedTarget)) setIsDragOver(false); };

  const handleBrainDrop = (e) => {
    e.preventDefault(); setIsDragOver(false);
    const paths = Array.from(e.dataTransfer.files)
      .map(f => rt.getPathForFile(f))
      .filter(p => p && (p.endsWith(".nii.gz") || p.endsWith(".nii")));
    if (paths.length) addBrainFilePaths(paths);
  };

  const handleCtDrop = (e) => {
    e.preventDefault(); setIsDragOver(false);
    const niftiPaths = Array.from(e.dataTransfer.files)
      .map(f => rt.getPathForFile(f))
      .filter(p => p && (p.endsWith(".nii.gz") || p.endsWith(".nii")));
    if (niftiPaths.length > 0) setFilePath(niftiPaths[0]);
  };

  const stopPoll = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };

  const startPolling = (job_id, case_id_local) => {
    setCaseId(case_id_local);
    setMessage("Segmentation started…");
    pollRef.current = setInterval(async () => {
      try {
        const s = await rt.uploadStatus(job_id);
        setProgress(s.progress || 0);
        setMessage(s.message || "");
        if (s.status === "done") {
          stopPoll();
          setPhase("done");
          setTimeout(() => { onClose(); onCaseReady(organ, s.case_id || case_id_local); }, 2000);
        } else if (s.status === "error") {
          stopPoll();
          setPhase("error");
          setMessage(s.error || "Inference failed.");
        }
      } catch {}
    }, 2000);
  };

  const startUpload = async () => {
    setPhase("running"); setProgress(0); setMessage("Uploading…");
    try {
      if (organ === "brain") {
        const paths = Object.values(brainFiles).filter(Boolean);
        const { job_id, case_id } = await rt.uploadInferBrain(paths);
        jobRef.current = job_id;
        startPolling(job_id, case_id);
      } else {
        const { job_id, case_id } = await rt.uploadInfer(filePath, organ);
        jobRef.current = job_id;
        startPolling(job_id, case_id);
      }
    } catch (e) {
      setPhase("error");
      setMessage(e?.message || String(e));
    }
  };

  useEffect(() => () => stopPoll(), []);

  const canStart = organ === "brain" ? !!brainFiles.flair : !!filePath;
  const seqCount = Object.keys(brainFiles).length;

  return (
    <div className="upload-modal-overlay" onClick={e => { if (e.target === e.currentTarget && phase !== "running") onClose(); }}>
      <div className="upload-modal">
        <div className="upload-modal__header">
          <Upload size={15}/> Upload &amp; Analyse New Case
          {phase !== "running" && (
            <button className="upload-modal__close" onClick={onClose}><X size={14}/></button>
          )}
        </div>

        {phase === "select" && (
          <>
            <div className="upload-modal__row">
              <label>Organ</label>
              <div className="upload-organ-group">
                {["brain","lung","liver"].map(o => (
                  <button key={o} type="button"
                    className={`upload-organ-btn${organ===o?" upload-organ-btn--on":""}`}
                    onClick={() => { setOrgan(o); setFilePath(""); setBrainFiles({}); }}>
                    {o === "lung" ? <LungIcon size={12}/> : o === "liver" ? <LiverIcon size={12}/> : <Brain size={12}/>}
                    {o.charAt(0).toUpperCase()+o.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {organ === "brain" ? (
              <div style={{padding:"10px 18px 0"}}>
                <div
                  className={`brain-drop-zone${isDragOver?" brain-drop-zone--over":""}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleBrainDrop}
                >
                  <Upload size={20} style={{color:"var(--accent)",opacity:0.75}}/>
                  <span>Drag &amp; drop <code>.nii.gz</code> files</span>
                  <button className="upload-browse-btn" onClick={chooseBrainFiles} style={{marginTop:4}}>
                    <FolderOpen size={12}/> Browse Files…
                  </button>
                </div>

                <div className="brain-seq-grid">
                  {BRAIN_SEQ_DEFS.map(({ key, label, required }) => (
                    <div key={key} className={
                      `brain-seq-slot` +
                      (brainFiles[key] ? " brain-seq-slot--ok" : "") +
                      (required && !brainFiles[key] ? " brain-seq-slot--req" : "")
                    }>
                      <span className="brain-seq-label">
                        {label}
                        {required && <span style={{color:"var(--danger)",marginLeft:2}}>*</span>}
                      </span>
                      {brainFiles[key] ? (
                        <>
                          <span className="brain-seq-file" title={brainFiles[key]}>
                            <BadgeCheck size={11} style={{color:"var(--success)",flexShrink:0}}/>
                            {brainFiles[key].split(/[\\/]/).pop()}
                          </span>
                          <button className="brain-seq-remove" onClick={() => removeBrainFile(key)} title="Remove">
                            <X size={10}/>
                          </button>
                        </>
                      ) : (
                        <span className="brain-seq-empty">Not selected</span>
                      )}
                    </div>
                  ))}
                </div>

                <div style={{display:"flex",alignItems:"center",gap:8,marginTop:7,fontSize:11,color:"var(--text-3)"}}>
                  {!brainFiles.flair && (
                    <span style={{display:"flex",alignItems:"center",gap:4}}>
                      <AlertTriangle size={11} style={{color:"var(--warn)"}}/> FLAIR required
                    </span>
                  )}
                  {seqCount > 0 && (
                    <span style={{marginLeft:"auto"}}>{seqCount}/4 sequences</span>
                  )}
                </div>
              </div>
            ) : (
              <div style={{padding:"10px 18px 0"}}>
                <div
                  className={`brain-drop-zone${isDragOver?" brain-drop-zone--over":""}${filePath?" brain-drop-zone--has-file":""}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleCtDrop}
                  onClick={choosFile}
                  style={{cursor:"pointer"}}
                >
                  {filePath ? (
                    <>
                      <BadgeCheck size={22} style={{color:"var(--success)"}}/>
                      <span style={{color:"var(--text-1)",fontWeight:600,fontSize:12}}>
                        {filePath.split(/[\\/]/).pop()}
                      </span>
                      <span style={{fontSize:10,color:"var(--text-3)"}}>Click or drop to replace</span>
                    </>
                  ) : (
                    <>
                      <Upload size={22} style={{color:"var(--accent)",opacity:0.75}}/>
                      <span>Drag &amp; drop <code>.nii.gz</code> or <code>.nii</code> file here</span>
                      <span style={{fontSize:11,color:"var(--text-3)"}}>or click to browse</span>
                    </>
                  )}
                </div>
                <div className="upload-file-row" style={{display:"none"}}>
                  <span className="upload-file-name">
                    {filePath ? filePath.split(/[\\/]/).pop() : "No file selected"}
                  </span>
                  <button className="upload-browse-btn" onClick={choosFile}>
                    <FolderOpen size={12}/> Browse…
                  </button>
                </div>
              </div>
            )}

            {canStart && (
              <div className="upload-modal__actions">
                <button className="upload-start-btn" onClick={startUpload}>
                  <Zap size={13}/> Start Analysis
                </button>
              </div>
            )}
          </>
        )}

        {phase === "running" && (
          <div className="upload-progress-wrap">
            <RefreshCw size={18} className="spin" style={{ color: "var(--accent)" }}/>
            <div className="upload-progress-label">{message}</div>
            <div className="upload-progress-bar">
              <div className="upload-progress-fill" style={{ width: `${progress}%` }}/>
            </div>
            <div className="upload-progress-pct">{progress}%</div>
            <p className="upload-progress-hint">Segmentation may take a few minutes. Do not close the window.</p>
          </div>
        )}

        {phase === "done" && (
          <div className="upload-done-wrap">
            <BadgeCheck size={32} style={{ color: "var(--success)" }}/>
            <p className="upload-done-title">Analysis complete!</p>
            <p className="upload-done-sub">Case ID: <strong>{caseId}</strong></p>
            <p style={{ fontSize: 11, color: "var(--text-3)", margin: "4px 0 10px" }}>
              Opening automatically…
            </p>
            <button className="upload-start-btn" onClick={() => { onClose(); onCaseReady(organ, caseId); }}>
              <Eye size={13}/> Open Now
            </button>
          </div>
        )}

        {phase === "error" && (
          <div className="upload-done-wrap">
            <AlertTriangle size={28} style={{ color: "var(--danger)" }}/>
            <p className="upload-done-title" style={{ color: "var(--danger)" }}>Analysis failed</p>
            <p className="upload-done-sub">{message}</p>
            <button className="upload-start-btn" onClick={() => setPhase("select")}>Retry</button>
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const[apiState,      setApiState]      =useState("checking");
  const[cases,         setCases]         =useState([]);
  const[lungCases,     setLungCases]     =useState([]);
  const[liverCases,    setLiverCases]    =useState([]);
  const[search,        setSearch]        =useState("");
  const[selectedId,    setSelectedId]    =useState("");
  const[selectedCase,  setSelectedCase]  =useState(null);
  const[activeTab,     setActiveTab]     =useState("report");
  const[question,      setQuestion]      =useState("");
  const[history,       setHistory]       =useState([]);
  const[qaSessionId,   setQaSessionId]   =useState(null);
  const[caseLoading,   setCaseLoading]   =useState(false);
  const[qaLoading,     setQaLoading]     =useState(false);
  const[reportBusy,    setReportBusy]    =useState(false);
  const[message,       setMessage]       =useState("");
  const[viewerModality,setViewerModality]=useState(null);
  const[viewerCaseId,  setViewerCaseId]  =useState(null);
  const[isDark,        setIsDark]        =useState(true);
  const[showUpload,    setShowUpload]    =useState(false);

  const[sidebarOpen,   setSidebarOpen]   =useState(true);
  const[inspectorOpen, setInspectorOpen] =useState(true);
  const[modalityFilter,setModalityFilter]=useState("all");
  const[caseSort,      setCaseSort]      =useState("default");
  const[highlightMode, setHighlightMode] =useState(false);
  const[showNotes,     setShowNotes]     =useState(false);
  const[reportNotes,   setReportNotes]   =useState("");
  const reportPaperRef=useRef(null);
  const feedEndRef=useRef(null);

  useEffect(()=>{
    document.documentElement.setAttribute("data-theme",isDark?"dark":"light");
    document.title = "CARVIS — Clinical AI Platform";
  },[isDark]);

  const sidebarSub=useMemo(()=>{
    if(activeTab==="viewer"){
      if(viewerModality==="lung")  return "Lung CT";
      if(viewerModality==="liver") return "Liver CT";
    }
    return "Clinical AI";
  },[activeTab,viewerModality]);

  const ctViewerActive=activeTab==="viewer"&&viewerCaseId&&viewerModality&&viewerModality!=="brain";
  const displayCaseId=ctViewerActive?viewerCaseId:selectedId;

  async function loadCase(id) {
    setSelectedId(id);setQuestion("");setHistory([]);
    const newSid = `${id}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2,7)}`;
    setQaSessionId(newSid);
    setCaseLoading(true);setMessage("");
    try{
      setSelectedCase(await getCaseDetail(id));
    }catch{
      setSelectedCase(null);
      toast.error("Case could not be loaded.");
      setMessage("Case could not be loaded.");
    }finally{setCaseLoading(false);}
  }

  async function refreshCtCases() {
    try{
      const[lung,liver]=await Promise.all([apiFetch("/ct-cases/lung"),apiFetch("/ct-cases/liver")]);
      setLungCases(lung||[]);setLiverCases(liver||[]);
    }catch{}
  }

  async function refreshBrainCases() {
    try{ const items=await getCases(); setCases(items); }catch{}
  }

  useEffect(()=>{
    let active=true;
    async function init(){
      for(let i=0;i<12&&active;i++){
        try{
          const items=await getCases();
          if(!active)return;
          setApiState("online");setCases(items);setMessage("");
          refreshCtCases();return;
        }catch{await new Promise(r=>setTimeout(r,900));}
      }
      if(active){setApiState("offline");setMessage("Cannot connect to the local API.");}
    }
    init();
    return()=>{active=false;};
  },[]);

  useEffect(()=>{feedEndRef.current?.scrollIntoView({behavior:"smooth"});},[history]);

  const _sortList=(arr,key="case_id")=>{
    if(caseSort==="az") return [...arr].sort((a,b)=>(a[key]||"").localeCompare(b[key]||""));
    if(caseSort==="za") return [...arr].sort((a,b)=>(b[key]||"").localeCompare(a[key]||""));
    return arr;
  };
  const filtered     =useMemo(()=>{const q=search.trim().toLowerCase();const mr=cases.filter(c=>(c.modality||"").toUpperCase()==="MR");const f=q?mr.filter(c=>[c.case_id,c.patient_id,c.study_id,c.modality].join(" ").toLowerCase().includes(q)):mr;return _sortList(f);},[cases,search,caseSort]);
  const filteredLung =useMemo(()=>{const q=search.trim().toLowerCase();const f=q?lungCases.filter(c=>c.case_id.toLowerCase().includes(q)):lungCases;return _sortList(f);},[lungCases,search,caseSort]);
  const filteredLiver=useMemo(()=>{const q=search.trim().toLowerCase();const f=q?liverCases.filter(c=>c.case_id.toLowerCase().includes(q)):liverCases;return _sortList(f);},[liverCases,search,caseSort]);

  const summary=useMemo(()=>{
    if(!selectedCase)return null;
    const tm=selectedCase.tumor_metrics||{};
    const cs=selectedCase.context?.segmentation_metrics||{};
    const nifti=selectedCase.findings_structured?.nifti_summary?.nifti_files||
                selectedCase.context?.findings_structured?.nifti_summary?.nifti_files||[];
    return{volume:tm.total_tumor_volume_mm3||cs.total_tumor_volume_mm3||0,lesions:tm.tumor_count||cs.tumor_count||0,diameter:tm.max_diameter_mm||cs.max_diameter_mm||0,sequences:nifti.map(humanSeq)};
  },[selectedCase]);

  async function submitQuestion(override){
    const q=(override||question).trim();
    if(!selectedId||!q||qaLoading)return;
    setQaLoading(true);setMessage("");setActiveTab("qa");
    try{
      const resp=await askQuestion(selectedId,q,qaSessionId);
      if(resp.session_id && resp.session_id !== qaSessionId) setQaSessionId(resp.session_id);
      const turnCount = resp.confidence_breakdown?.session_turns_used ?? 0;
      setHistory(prev=>[...prev,{
        id:`${Date.now()}-${Math.random().toString(16).slice(2)}`,
        question:q,
        ...resp,
        status:answerStatus(resp),
        sessionTurn: turnCount,
      }]);
      setQuestion("");
    }catch{setMessage("Question could not be processed.");}
    finally{setQaLoading(false);}
  }

  async function refreshReport(){
    if(!selectedId||reportBusy)return;
    setReportBusy(true);setMessage("");
    const tid = toast.loading("Regenerating report…");
    try{
      const resp=await regenerateReport(selectedId);
      setSelectedCase(prev=>({...prev,report_markdown:resp.report_markdown}));
      setActiveTab("report");
      toast.success("Report updated.", { id: tid });
    }catch{
      toast.error("Report could not be regenerated.", { id: tid });
      setMessage("Report could not be regenerated.");
    }finally{setReportBusy(false);}
  }

  function openCtCase(modality,id){
    setViewerModality(modality);
    setViewerCaseId(id);
    setSelectedId(id);
    setSelectedCase(null);
    setHistory([]);
    setQuestion("");
    setMessage("");
    setActiveTab("viewer");
    const newSid = `${id}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2,7)}`;
    setQaSessionId(newSid);
    getCaseDetail(id).then(detail=>setSelectedCase(detail)).catch(()=>{});
  }

  const handleReportClick=useCallback(e=>{
    if(!highlightMode)return;
    const el=e.target.closest('p,li,h1,h2,h3,h4,blockquote,td');
    if(!el)return;
    if(el.hasAttribute('data-hl')) el.removeAttribute('data-hl');
    else el.setAttribute('data-hl','1');
  },[highlightMode]);

  const handlePrint=()=>{
    downloadReportAsPdf(displayCaseId, selectedCase?.report_markdown, reportPaperRef.current, reportNotes);
  };
  const handleSave=()=>{
    saveReportToFile(displayCaseId, selectedCase?.report_markdown, reportPaperRef.current, reportNotes);
  };

  const InspectorPanel = () => {
    if (!selectedCase && !viewerCaseId) return (
      <div className="inspector-empty">
        <Info size={18} style={{color:"var(--text-3)"}}/>
        <p>Select a case to view details</p>
      </div>
    );
    const tm = selectedCase?.tumor_metrics || {};
    const burden = tm.tumor_burden || {};
    const modLabel = viewerModality === "lung" ? "Lung CT"
      : viewerModality === "liver" ? "Liver CT" : "Brain MRI";
    const cid = displayCaseId || viewerCaseId;
    return (
      <div className="inspector-content">
        <div className="inspector-section">
          <div className="inspector-section__title">Case</div>
          <div className="inspector-row"><span>ID</span><strong className="inspector-val">{cid || "—"}</strong></div>
          <div className="inspector-row"><span>Modality</span><strong className="inspector-val">{modLabel}</strong></div>
          {selectedCase?.modality && <div className="inspector-row"><span>Sequence</span><strong className="inspector-val">{selectedCase.modality}</strong></div>}
        </div>
        {(burden.total_volume_mm3 !== undefined) && (
          <div className="inspector-section">
            <div className="inspector-section__title">Tumor Burden</div>
            <div className="inspector-row"><span>Total Volume</span><strong className="inspector-val">{burden.total_volume_mm3 ? `${(burden.total_volume_mm3/1000).toFixed(2)} cm³` : "—"}</strong></div>
            <div className="inspector-row"><span>Max Diameter</span><strong className="inspector-val">{burden.max_diameter_mm ? `${burden.max_diameter_mm.toFixed(1)} mm` : "—"}</strong></div>
            {burden.enhancing_volume_mm3 !== undefined && <div className="inspector-row"><span>Enhancing</span><strong className="inspector-val">{(burden.enhancing_volume_mm3/1000).toFixed(2)} cm³</strong></div>}
            {burden.edema_volume_mm3 !== undefined && <div className="inspector-row"><span>Edema</span><strong className="inspector-val">{(burden.edema_volume_mm3/1000).toFixed(2)} cm³</strong></div>}
            {burden.necrotic_volume_mm3 !== undefined && <div className="inspector-row"><span>Necrotic</span><strong className="inspector-val">{(burden.necrotic_volume_mm3/1000).toFixed(2)} cm³</strong></div>}
          </div>
        )}
        <div className="inspector-section">
          <div className="inspector-section__title">Quick Actions</div>
          <button className="inspector-action" onClick={()=>setActiveTab("report")}><FileText size={12}/> View Report</button>
          <button className="inspector-action" onClick={()=>setActiveTab("qa")}><MessageSquare size={12}/> Clinical Q&A</button>
          <button className="inspector-action" onClick={()=>setActiveTab("viewer")}><Scan size={12}/> Open Imaging</button>
          {selectedCase?.report_markdown && (
            <button className="inspector-action inspector-action--secondary" onClick={handlePrint}><Printer size={12}/> Print / PDF</button>
          )}
        </div>
        <div className="inspector-section">
          <div className="inspector-section__title">Status</div>
          <div className="inspector-row">
            <span>Report</span>
            <span className={`inspector-badge ${selectedCase?.report_markdown?"inspector-badge--ok":"inspector-badge--warn"}`}>
              {selectedCase?.report_markdown ? "Ready" : "Not generated"}
            </span>
          </div>
          <div className="inspector-row">
            <span>Q&A History</span>
            <span className="inspector-badge inspector-badge--info">{history.length} turn{history.length!==1?"s":""}</span>
          </div>
        </div>
      </div>
    );
  };

  return(
    <TooltipProvider>
    <div className={`shell${sidebarOpen?"":" shell--collapsed"}${inspectorOpen?" shell--inspector":""}`}>
      <aside className={`sidebar${sidebarOpen?"":" sidebar--collapsed"}`}>
        {sidebarOpen ? (
          <div className="sidebar-brand">
            <div className="sidebar-icon"><CarvisLogoSmall /></div>
            <div style={{flex:1}}>
              <div className="sidebar-name" style={{fontFamily:"var(--font-ui)",letterSpacing:"0.04em"}}>CARVIS</div>
              <div className="sidebar-sub">{sidebarSub}</div>
            </div>
            <Tip content={isDark?"Light mode":"Dark mode"} side="right">
              <button type="button" className="theme-toggle" onClick={()=>{
                const next=!isDark;
                document.documentElement.setAttribute("data-theme",next?"dark":"light");
                setIsDark(next);
              }}>
                {isDark?<Sun size={13}/>:<Moon size={13}/>}
              </button>
            </Tip>
          </div>
        ) : (
          <div className="sidebar-mini">
            <div className="sidebar-icon" style={{marginBottom:8}}><CarvisLogoSmall /></div>
            <button type="button" className="sidebar-mini-toggle" onClick={()=>setSidebarOpen(true)} title="Expand sidebar">
              <ChevronRight size={14}/>
            </button>
          </div>
        )}

        {sidebarOpen&&<div className="sidebar-search">
          <Search size={12}/>
          <input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search cases…"/>
        </div>}

        {sidebarOpen&&<div className="sidebar-filter-row">
          {[{key:"all",label:"All"},
            {key:"brain",label:"Brain",icon:<Brain size={9}/>},
            {key:"lung",label:"Lung",icon:<LungIcon size={9}/>},
            {key:"liver",label:"Liver",icon:<LiverIcon size={9}/>}
          ].map(f=>(
            <button key={f.key} type="button"
              className={`mod-filter-chip${modalityFilter===f.key?" mod-filter-chip--active":""} ${f.key!=="all"?`mod-filter-chip--${f.key}`:""}`}
              onClick={()=>setModalityFilter(f.key)}>
              {f.icon&&<span style={{marginRight:2}}>{f.icon}</span>}{f.label}
            </button>
          ))}
          <Tip content={caseSort==="default"?"Sort A→Z":caseSort==="az"?"Sort Z→A":"Default order"} side="right">
            <button type="button" className={`sort-toggle-btn${caseSort!=="default"?" sort-toggle-btn--active":""}`}
              onClick={()=>setCaseSort(s=>s==="default"?"az":s==="az"?"za":"default")}>
              {caseSort==="za"?<ArrowUpAZ size={11}/>:<ArrowDownAZ size={11}/>}
            </button>
          </Tip>
        </div>}

        <div className="sidebar-scroll" style={{display:sidebarOpen?"flex":"none",flexDirection:"column"}}>
          {(modalityFilter==="all"||modalityFilter==="brain")&&<>
          <div className="sidebar-label sidebar-label--brain">
            <Brain size={10}/> Brain <span className="label-line"/>
            <span className="sidebar-label-count">{filtered.length}</span>
            <button type="button" onClick={refreshBrainCases} style={{background:"none",border:"none",cursor:"pointer",padding:"0 2px",color:"var(--blue-bright)"}}>
              <RefreshCw size={9}/>
            </button>
          </div>
          <div className="case-list">
            {filtered.map(c=>(
              <button key={c.case_id} type="button"
                className={`case-item${selectedId===c.case_id&&!ctViewerActive?" case-item--active":""}`}
                onClick={()=>{loadCase(c.case_id);setViewerModality("brain");setViewerCaseId(null);}}>
                <div className="case-item__row">
                  <span className="case-item__id">{c.case_id}</span>
                  {selectedId===c.case_id&&!ctViewerActive&&<span className="case-item__dot case-item__dot--brain"/>}
                </div>
              </button>
            ))}
            {!filtered.length&&<div className="case-list-empty">No brain cases.</div>}
          </div>
          </>}

          {(modalityFilter==="all"||modalityFilter==="lung")&&<>
          <div className="sidebar-label sidebar-label--lung">
            <LungIcon size={10}/> Lung <span className="label-line"/>
            <span className="sidebar-label-count">{filteredLung.length}</span>
            <button type="button" onClick={refreshCtCases} style={{background:"none",border:"none",cursor:"pointer",padding:"0 2px",color:"var(--lung)"}}>
              <RefreshCw size={9}/>
            </button>
          </div>
          <div className="case-list">
            {filteredLung.map(c=>(
              <button key={c.case_id} type="button"
                className={`case-item case-item--lung${viewerCaseId===c.case_id&&viewerModality==="lung"?" case-item--lung-active":""}`}
                onClick={()=>openCtCase("lung",c.case_id)}>
                <div className="case-item__row">
                  <span className="case-item__id">{c.case_id}</span>
                  {viewerCaseId===c.case_id&&viewerModality==="lung"&&<span className="case-item__dot case-item__dot--lung"/>}
                </div>
              </button>
            ))}
            {!filteredLung.length&&<div className="case-list-empty">{lungCases.length===0?"Add files to lung_data/":"No matches."}</div>}
          </div>
          </>}

          {(modalityFilter==="all"||modalityFilter==="liver")&&<>
          <div className="sidebar-label sidebar-label--liver">
            <LiverIcon size={10}/> Liver <span className="label-line"/>
            <span className="sidebar-label-count">{filteredLiver.length}</span>
            <button type="button" onClick={refreshCtCases} style={{background:"none",border:"none",cursor:"pointer",padding:"0 2px",color:"var(--liver)"}}>
              <RefreshCw size={9}/>
            </button>
          </div>
          <div className="case-list">
            {filteredLiver.map(c=>(
              <button key={c.case_id} type="button"
                className={`case-item case-item--liver${viewerCaseId===c.case_id&&viewerModality==="liver"?" case-item--liver-active":""}`}
                onClick={()=>openCtCase("liver",c.case_id)}>
                <div className="case-item__row">
                  <span className="case-item__id">{c.case_id}</span>
                  {viewerCaseId===c.case_id&&viewerModality==="liver"&&<span className="case-item__dot case-item__dot--liver"/>}
                </div>
              </button>
            ))}
            {!filteredLiver.length&&<div className="case-list-empty">{liverCases.length===0?"Add files to liver_data/":"No matches."}</div>}
          </div>
          </>}
        </div>

        {sidebarOpen&&<div className="sidebar-upload-btn-wrap">
          <button type="button" className="sidebar-upload-btn" onClick={()=>setShowUpload(true)}>
            <Upload size={12}/> Upload New Case
          </button>
        </div>}

        {sidebarOpen&&<div className="sidebar-footer">
          <span className={`status-dot status-dot--${apiState}`}/>
          <span>{apiState==="online"?"Connected":apiState==="offline"?"Offline":"Connecting…"}</span>
          <button type="button" className="sidebar-collapse-btn-inline" onClick={()=>setSidebarOpen(false)} title="Collapse sidebar">
            <ChevronLeft size={12}/>
          </button>
        </div>}
      </aside>

      <main className="workspace">
        {selectedId ? (
          <>
            <div className="case-bar">
              <div className="case-bar__left">
                <h1 className="case-bar__id">
                  {displayCaseId}
                  {ctViewerActive&&<span className={`case-bar__modality-badge${viewerModality==="lung"?" case-bar__modality-badge--lung":""}`}>{viewerModality?.toUpperCase()} CT</span>}
                </h1>
                {summary&&!ctViewerActive&&(
                  <div className="case-bar__meta">
                    {summary.sequences.length>0&&<><span>{summary.sequences.join(" · ")}</span><span className="meta-dot"/></>}
                    <span>{toCm3(summary.volume)} burden</span>
                    <span className="meta-dot"/>
                    <span>{summary.lesions} lesion{summary.lesions!==1?"s":""}</span>
                    <span className="meta-dot"/>
                    <span>Max Ø {summary.diameter} mm</span>
                  </div>
                )}
              </div>
              <div className="case-bar__right">
                {activeTab==="report"&&selectedCase?.report_markdown&&(
                  <>
                    <button className="btn-pdf" onClick={handlePrint} title="Print or save as PDF">
                      <Printer size={13}/> Print / PDF
                    </button>
                    <button className="btn-save" onClick={handleSave} title="Save report as PDF to PC">
                      <Download size={13}/> Save to PC
                    </button>
                  </>
                )}
                <button className="btn-refresh" onClick={refreshReport} disabled={reportBusy}>
                  <RefreshCw size={12} className={reportBusy?"spin":""}/> Refresh
                </button>
              </div>
            </div>

            <div className="tab-bar">
              {[{key:"report",icon:<FileText size={13}/>,label:"Report"},{key:"qa",icon:<MessageSquare size={13}/>,label:"Clinical Q&A"},{key:"viewer",icon:<Layers size={13}/>,label:"Imaging"}]
                .map(t=>(
                  <button key={t.key} type="button"
                    className={`tab-btn${activeTab===t.key?" tab-btn--active":""}`}
                    onClick={()=>setActiveTab(t.key)}>
                    {t.icon} {t.label}
                  </button>
                ))}
            </div>

            <div className="content-pane">
              {activeTab==="report"&&(
                caseLoading?(
                  <div className="pane-center"><RefreshCw size={20} className="spin" style={{color:"var(--blue)"}}/><p>Loading…</p></div>
                ):selectedCase?.report_markdown?(
                  <div className="report-scroll">
                    <div className="report-annotation-bar">
                      <button className="report-action-btn" onClick={handlePrint}>
                        <Printer size={13}/> Print / Save as PDF
                      </button>
                      <button className="report-action-btn report-action-btn--save" onClick={handleSave}>
                        <Download size={13}/> Save to PC
                      </button>
                      <div className="report-annotation-bar__sep"/>
                      <button
                        className={`report-annot-btn${highlightMode?" report-annot-btn--on":""}`}
                        onClick={()=>setHighlightMode(v=>!v)}>
                        <Highlighter size={13}/> {highlightMode?"Exit Highlight":"Highlight"}
                      </button>
                      {highlightMode&&<span className="report-hl-hint">Click any paragraph to highlight · Click again to remove</span>}
                      <div style={{flex:1}}/>
                      <button
                        className={`report-annot-btn${showNotes?" report-annot-btn--on":""}`}
                        onClick={()=>setShowNotes(v=>!v)}>
                        <StickyNote size={13}/> Notes {showNotes?<ChevronUp size={11}/>:<ChevronDown size={11}/>}
                      </button>
                    </div>
                    {showNotes&&(
                      <div className="report-notes-panel">
                        <div className="report-notes-panel__header">
                          <StickyNote size={13}/>
                          <span>Clinical Notes</span>
                          <span style={{fontSize:10,color:"var(--text-3)",marginLeft:"auto"}}>Included in saved output</span>
                          <button className="report-notes-close" onClick={()=>setShowNotes(false)}><X size={12}/></button>
                        </div>
                        <textarea
                          className="report-notes-textarea"
                          value={reportNotes}
                          onChange={e=>setReportNotes(e.target.value)}
                          placeholder="Add clinical observations, notes, or follow-up actions here…"
                          rows={4}
                        />
                      </div>
                    )}
                    <article
                      className={`report-paper${highlightMode?" report-paper--hl-mode":""}`}
                      ref={reportPaperRef}
                      onClick={handleReportClick}>
                      <ReactMarkdown components={reportMarkdownComponents}>{selectedCase.report_markdown}</ReactMarkdown>
                    </article>
                    {highlightMode&&<div className="report-hl-tip">Highlight mode active · Click elements to mark · Click again to remove</div>}
                  </div>
                ):(
                  <div className="pane-center">
                    <FileText size={26} style={{color:"var(--muted)"}}/>
                    <p style={{color:"var(--text-2)"}}>No report available.</p>
                    <button className="btn-refresh" onClick={refreshReport} disabled={reportBusy}>Generate Report</button>
                  </div>
                )
              )}

              {activeTab==="qa"&&(
                <div className="qa-pane">
                  <div className="safety-bar">
                    <ShieldAlert size={12}/>
                    Decision-support only. Not a medical diagnosis or treatment recommendation.
                  </div>
                  <div className="qa-feed">
                    {history.length===0&&(
                      <div className="qa-feed-empty">
                        <div className="qa-empty-icon"><MessageSquare size={24}/></div>
                        <p>Ask a clinical question about <strong>{selectedId}</strong></p>
                      </div>
                    )}
                    {history.map(entry=>(
                      <div key={entry.id} className="qa-exchange">
                        <div className="qa-question-wrap">
                          <div className="qa-question-bubble">{entry.question}</div>
                        </div>
                        <div className={`qa-answer-card qa-answer-card--${entry.status}`}>
                          <div className="qa-answer-card__header">
                            <span className={`qa-status-badge qa-status-badge--${entry.status}`}>
                              {entry.status==="supported"?<><BadgeCheck size={10}/> Supported</>
                               :entry.status==="insufficient"?<><AlertTriangle size={10}/> Insufficient</>
                               :<><AlertTriangle size={10}/> Low evidence</>}
                            </span>
                            <div className="qa-confidence-bar">
                              <div className="qa-confidence-bar__fill" style={{width:`${Math.round(entry.confidence*100)}%`,background:entry.status==="supported"?"var(--success)":entry.status==="insufficient"?"var(--warn)":"var(--text-4)"}}/>
                            </div>
                            <span className="qa-confidence-pct">{Math.round(entry.confidence*100)}%</span>
                          </div>
                          {entry.status==="insufficient"&&(
                            <div className="qa-answer-card__alert">Evidence was insufficient for a confident answer.</div>
                          )}
                          <p className="qa-answer-card__body">{entry.answer}</p>
                          {entry.evidence_ids.length>0&&(
                            <div className="qa-answer-card__evidence">
                              <span className="qa-evidence-label">{entry.evidence_ids.length} source{entry.evidence_ids.length!==1?"s":""}</span>
                              {entry.evidence_ids.map(id=><code key={id} className="ev-tag">{id}</code>)}
                            </div>
                          )}
                          {entry.safety_note&&<p className="qa-answer-card__safety">{entry.safety_note}</p>}
                          {entry.clinical_implication&&(()=>{
                            const ci=entry.clinical_implication;
                            const tier=ci.overall_tier||"";
                            const isDup=ci.summary&&entry.answer.includes(ci.summary.slice(0,40));
                            return(
                              <div className={`qa-implication qa-implication--${tier}`}>
                                <span className="qa-tier-badge">{tier.replace("_"," ").toUpperCase()}</span>
                                {!isDup&&ci.summary&&<span className="qa-implication__summary">{ci.summary}</span>}
                                {ci.items?.map((item,i)=>(
                                  <div key={i} className="qa-implication__item">
                                    <span className={`qa-tier-mini qa-tier-mini--${item.tier||"none"}`}>{item.tier}</span>
                                    <span>{item.rationale}</span>
                                  </div>
                                ))}
                              </div>
                            );
                          })()}
                          {entry.supporting_cases?.length>0&&(
                            <div className="qa-supporting">
                              <span className="qa-evidence-label">Supporting cases</span>
                              {entry.supporting_cases.map(c=>(
                                <span key={c.case_id} className="qa-supporting__case" title={c.similarity_reason}>
                                  {c.case_id}
                                </span>
                              ))}
                            </div>
                          )}
                          {(entry.intent_class||entry.retrieval_count>0)&&(
                            <div className="qa-meta-row">
                              {entry.intent_class&&(
                                <span className="qa-meta-badge qa-meta-badge--intent">
                                  <Zap size={9}/> {entry.intent_class}
                                </span>
                              )}
                              {entry.retrieval_count>0&&(
                                <span className="qa-meta-badge qa-meta-badge--retrieval">
                                  <Activity size={9}/> {entry.retrieval_count} sources
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                    <div ref={feedEndRef}/>
                  </div>
                  <div className="qa-prompts">
                    {(ctViewerActive ? CT_PROMPTS : BRAIN_PROMPTS).map(p=>(
                      <button key={p} type="button" className="prompt-btn" disabled={qaLoading}
                        onClick={()=>submitQuestion(p)}>{p}</button>
                    ))}
                  </div>
                  <div className="composer">
                    {qaLoading&&<div className="composer-typing"><span/><span/><span/></div>}
                    <textarea value={question} onChange={e=>setQuestion(e.target.value)}
                      placeholder="Ask a clinical question… (Enter to send)"
                      disabled={qaLoading} rows={3}
                      onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey&&question.trim()){e.preventDefault();submitQuestion();}}}/>
                    <div className="composer__footer">
                      <span className="composer__hint" style={{color:"var(--text-3)",fontSize:11}}>Locked to {selectedId}</span>
                      <button type="button" className="btn-send"
                        onClick={()=>submitQuestion()} disabled={!question.trim()||qaLoading}>
                        <Send size={12}/>{qaLoading?"Asking…":"Send"}
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {activeTab==="viewer"&&(
                <ViewerPane
                  key={`${viewerModality||"brain"}::${viewerCaseId||selectedId}`}
                  caseId={selectedId}
                  defaultModality={viewerModality}
                  defaultCaseId={viewerCaseId}
                />
              )}
            </div>
          </>
        ):(
          <WelcomeScreen brainCount={filtered.length} lungCount={lungCases.length} liverCount={liverCases.length}/>
        )}
        {message&&<div className="message-bar">{message}</div>}
      </main>

      <aside className={`inspector-panel${inspectorOpen?" inspector-panel--open":""}`}>
        <div className="inspector-header">
          <span className="inspector-title"><Info size={12}/> Inspector</span>
          <Tip content="Close panel" side="left">
            <button className="inspector-close" onClick={()=>setInspectorOpen(false)}>
              <PanelRightClose size={13}/>
            </button>
          </Tip>
        </div>
        <InspectorPanel/>
      </aside>

      {!inspectorOpen && (
        <Tip content="Open inspector" side="left">
          <button className="inspector-toggle-btn" onClick={()=>setInspectorOpen(true)}>
            <PanelRight size={14}/>
          </button>
        </Tip>
      )}

      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onCaseReady={(organ, id) => {
            if(organ === "brain") {
              refreshBrainCases();
              setViewerModality("brain"); setViewerCaseId(null);
              loadCase(id);
              toast.success(`Brain case ${id} loaded.`);
            } else {
              refreshCtCases();
              openCtCase(organ, id);
              toast.success(`${organ.charAt(0).toUpperCase()+organ.slice(1)} case ${id} loaded.`);
            }
            setShowUpload(false);
          }}
        />
      )}

      <Toaster
        position="bottom-right"
        theme="dark"
        toastOptions={{
          style: { background: "#111e32", border: "1px solid #1e2f48", color: "#e2e8f0", fontSize: 13 },
        }}
      />
    </div>
    </TooltipProvider>
  );
}
