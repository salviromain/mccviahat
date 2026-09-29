import fs from 'node:fs/promises';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';
import {resolvePresentationFont,finalizePresentation} from '/Users/rsalvi/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='/Users/rsalvi/Desktop/mccviahat';
const skill='/Users/rsalvi/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const tmp=root+'/tmp/pipeline_slide';
const font=resolvePresentationFont();
const p=Presentation.create({slideSize:{width:1600,height:900}});
const s=p.slides.add();s.background.fill='#FAFBFC';
const ink='#142D42',muted='#496173',teal='#007F83',blue='#255CAD';
function text(t,x,y,w,h,size=25,color=ink,bold=false){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={typeface:font,fontSize:size,color,bold,autoFit:'none'};return a;}
text('Pipeline sperimentale HAT',60,38,1450,66,49,ink,true);
text('Il contenuto affettivo dei prompt si associa a differenze misurabili nei segnali hardware?',60,111,1480,55,29,muted);
const xs=[60,360,660,960,1260];
const heads=['Prompt','Inferenza LLM','Acquisizione HAT','Feature temporali','Analisi'];
const bodies=[
'20 emotivi + 20 neutrali\nLunghezze in token\napprossimativamente\nbilanciate',
'Llama-2 7B\nLlama-3.1 70B\nQ4_K_M, llama.cpp\nDocker, output 50 token',
'perf stat: 1 ms nominale\nContatori OS: 100 ms\ncore_power.throttle\nTracce CSV + metadati',
'Media per unità di tempo\nPendenza e varianza\nEntropia spettrale\nComplessità Lempel–Ziv',
'Confronto tra condizioni\nk-means (k = 2)\nTest + Bonferroni\nEffetti e robustezza'
];
let anchors=[];
for(let i=0;i<5;i++){
 text(String(i+1).padStart(2,'0'),xs[i],193,90,48,32,teal,true);
 anchors.push(text(heads[i],xs[i],250,276,53,29,ink,true));
 text(bodies[i],xs[i],319,280,172,24,muted);
}
for(let i=0;i<4;i++)s.shapes.connect(anchors[i],anchors[i+1],{kind:'straight',fromSide:'right',toSide:'left',line:{fill:'#91A6B5',width:2},tail:{type:'arrow',width:'sm',length:'sm'}});
text('Ambiente di esecuzione',60,491,340,40,25,ink,true);
text('CloudLab bare-metal dedicato, dual Xeon Gold 6142, Ubuntu 24.04',405,491,1135,40,25,muted);
text('Per-trial',60,570,230,46,29,blue,true);
text('Reset del container tra prompt, flush della cache OS e pausa di stabilizzazione',305,567,1235,43,25,ink);
text('Una traccia per inferenza. 8 run × 40 trial = 320 osservazioni per modello',305,609,1235,42,24,muted);
text('Full-trace',60,677,230,46,29,teal,true);
text('20 prompt consecutivi, senza reset intermedi. Un’unica acquisizione continua',305,674,1235,43,25,ink);
text('Una traccia per fase. Nel paper: 20 tracce per condizione sul 70B',305,716,1235,42,24,muted);
text('Entrambi i protocolli prevedono un reboot del nodo tra le fasi emotiva e neutrale.',60,778,1480,37,24,ink,true);
text('In corso: condizione joyful e controlli sul workload mediante istruzioni, cicli e IPC.',60,823,1480,37,23,teal);
s.speakerNotes.textFrame.setText(`Pipeline del progetto mccviahat, con distinzione fra protocolli del paper e sviluppi in corso.\nFonti: /Users/rsalvi/Downloads/AGI_2026_SalviWolfson_paper122 to_appear.pdf, sezioni 2–5 e appendice B. Repository: scripts/run/run_prompts_isolated.py, scripts/run/run_prompts_fulltrace.py, scripts/collectors/substrate_collector.py, scripts/run/extract_features.py, scripts/run/extract_features_fulltrace.py, analysis/notebooks/.\nHAT = Hardware Anomaly Trace, serie temporali degli indicatori hardware e OS.\n1. 1 ms è l'intervallo richiesto a perf, non una garanzia di cadenza effettiva uniforme. Il collector usa misure system-wide. Il PID serve anche al monitoraggio del processo. CORE_POWER.THROTTLE va interpretato secondo la definizione dell'evento della CPU, non come sinonimo generico di temperatura o PROCHOT.\n2. Le cinque feature elencate sono quelle del paper. L'estrattore full-trace attuale e il relativo notebook usano una selezione diversa e non includono LZ.\n3. Nel paper il clustering k-means e il filtro di coerenza direzionale ≥6/8 run riguardano il per-trial. I test confrontano le feature fra condizioni: MWU nel per-trial, MWU e Welch nel full-trace. Bonferroni segue la famiglia di confronti dichiarata. Gli sweep sui prompt e i null test sono controlli descrittivi di robustezza, non stime di generalizzazione su nuovi dati.\nNei per-trial: 20 prompt per condizione ripetuti su 8 run. I trial condividono prompt e sessione e non equivalgono a 320 unità sperimentali indipendenti. Nei full-trace una fase di 20 prompt fornisce un'osservazione. Il codice permette anche segmentazione per prompt tramite i timestamp, mantenendo la dipendenza entro traccia.\nSviluppi in corso: dataset joyful e misure di istruzioni/cicli per IPC. I metadati di quattro cartelle *_N_1 riportano joy: risolvere la provenienza prima di interpretare confronti su quelle acquisizioni.\nLa slide descrive il flusso sperimentale. Le differenze hardware non dimostrano da sole esperienza soggettiva.`);
await (await PresentationFile.exportPptx(p)).save(tmp+'/candidate.pptx');
const final=root+'/output/presentations/pipeline_mccviahat_final.pptx';
await finalizePresentation({workspaceDir:root,candidatePath:tmp+'/candidate.pptx',finalPath:final,pythonExecutable:'/Users/rsalvi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:tmp+'/validation_final.json'});
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(final));
const png=await finalDeck.export({slide:finalDeck.slides.items[0],format:'png',scale:1});
await fs.writeFile(root+'/output/presentations/pipeline_mccviahat.png',new Uint8Array(await png.arrayBuffer()));
console.log(final);
