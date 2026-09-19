'use strict';
const $ = id => document.getElementById(id);
const state = {records: [], task: null, offset: 0, config: null, dirty: false, scene: {}, draw: null, points: [], loading: false, clipStart: 0};
const labels = {QUEUED:'排队中',PROCESSING:'分析中',ANALYZED:'分析完成',REJECTED:'未检出 / 无效',ERROR:'处理失败',EXPIRED:'已过期',
  UNKNOWN:'无法可靠判定',CANDIDATE:'发现疑似候选',CONFIRMED:'人工确认',VALID:'人工确认',INVALID:'人工标记无效',UNCERTAIN:'证据不足',RESET:'恢复 AI 结果',
  NONE:'未发现违法',SOLID_LINE:'疑似压实线',LATERAL_MOVEMENT:'横向移动（待复核）',WRONG_WAY:'疑似逆行',RED_LIGHT:'疑似闯红灯',RESTRICTED_LANE:'疑似占用非机动车道',
  RED:'红灯',GREEN:'绿灯',YELLOW:'黄灯',OFF:'未见到灯',
  NEEDS_CALIBRATION:'缺少固定机位与道路标定',MOVING_CAMERA:'画面移动，暂停几何规则',INSUFFICIENT_BACKGROUND:'背景特征不足，无法判断机位',
  RED_LIGHT_CANDIDATE:'红灯稳定且前车继续接近，仅作候选',
  PLATE_UNCONFIRMED:'未确认车牌，已停止违法判定',
  VIOLATION_RULES_NOT_IMPLEMENTED:'旧版本未启用违法规则',HUMAN_INVALIDATED:'人工标记无效',HUMAN_UNCERTAIN:'人工标记证据不足',HUMAN_CONFIRMED:'人工确认 / 纠正',
  manual:'按钮标记',voice:'语音标记',automatic:'自动筛查',import:'导入视频',car:'汽车',truck:'卡车',bus:'公交车',motorcycle:'摩托车'};
const title = v => labels[v] || v || '—';
function plateTrust(plate){return (plate.stable?1e6:0)+(plate.hits||0)*1e3+(plate.confidence||0)*100;}
function plateHeat(t){
  const mix=(a,b,u)=>a.map((x,i)=>Math.round(x+(b[i]-x)*u));
  const rgb=t>=.5?mix([46,168,94],[232,61,48],(t-.5)*2):mix([255,255,255],[46,168,94],t*2);
  const lum=(0.2126*rgb[0]+0.7152*rgb[1]+0.0722*rgb[2])/255;
  return {bg:`rgb(${rgb.join(',')})`,fg:lum>0.62?'#173042':'#fff'};
}
const date = seconds => new Date(seconds*1000).toLocaleString('zh-CN',{hour12:false});
function element(tag, text, className) { const node=document.createElement(tag); if(text!==undefined)node.textContent=text; if(className)node.className=className; return node; }
function toast(message) { $('toast').textContent=message; $('toast').hidden=false; clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').hidden=true,4500); }
async function sha256hex(text){
  const buf=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map(b=>b.toString(16).padStart(2,'0')).join('');
}
let session='', sessionPromise=null;
let mediaRetry=false;
async function ensureSession(){
  if(session)return;
  if(sessionPromise)return sessionPromise;
  sessionPromise=(async()=>{
    const ts=Math.floor(Date.now()/1000),nonce=crypto.randomUUID().replaceAll('-','');
    let device;
    try{device=localStorage.getItem('traffic-device-id');}catch{}
    if(!device)device=crypto.randomUUID();
    const platform='web';
    const code=await sha256hex(`${device}\n${platform}\n${ts}\n${nonce}\ntraffic-hello-v1`);
    const response=await fetch('/v1/hello',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify({device_id:device,platform,model:(navigator.userAgent||'web').slice(0,120),app_version:'2.0',ts,nonce,code})});
    const data=await response.json();
    if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'握手失败');
    session=data.session;
    try{localStorage.setItem('traffic-device-id',data.device_id||device);}catch{}
  })();
  try{await sessionPromise;}finally{sessionPromise=null;}
}
async function api(path, options={}, retried=false) {
  await ensureSession();
  const headers={'X-Requested-With':'traffic-console',Authorization:'Bearer '+session,...(options.headers||{})};
  const next={...options};
  if(next.json!==undefined){next.body=JSON.stringify(next.json);headers['Content-Type']='application/json';delete next.json;}
  const response=await fetch(path,{...next,headers,credentials:'same-origin'});
  if(response.status===401 && !retried){session='';return api(path,options,true);}
  const data=await response.json();
  if(!response.ok){const detail=typeof data.detail==='string'?data.detail:JSON.stringify(data.detail);throw new Error(detail||`请求失败 ${response.status}`);}
  return data;
}
function run(action){return async event=>{event?.preventDefault();try{await action(event);}catch(error){toast(error.message);}};}
function badge(value){const node=element('span',title(value),'pill');if(['ANALYZED','VALID','CONFIRMED'].includes(value))node.classList.add('good');else if(['ERROR','INVALID'].includes(value))node.classList.add('bad');else if(['QUEUED','PROCESSING','CANDIDATE'].includes(value))node.classList.add('warn');return node;}
function fields(node, values){node.replaceChildren();for(const [key,value] of values){node.append(element('dt',key),element('dd',value??'—'));}}

async function refresh(){
  if(state.loading||$('loginDialog').open)return;
  state.loading=true;
  try{
    const query=new URLSearchParams({limit:20,offset:state.offset,status:$('statusFilter').value,review:$('reviewFilter').value,query:$('search').value.trim()});
    const [list,overview]=await Promise.all([api('/v1/tasks?'+query),api('/v1/overview')]);
    state.records=list.tasks;renderRecords();
    $('countTotal').textContent=overview.total;
    $('countProcessing').textContent=(overview.counts.QUEUED||0)+(overview.counts.PROCESSING||0);
    $('countDone').textContent=overview.counts.ANALYZED||0;$('countReview').textContent=overview.intervened;
    const clients=overview.clients||[];
    $('countClients').textContent=overview.client_count??clients.length;
    $('clientList').textContent=clients.slice(0,8).map(c=>`${c.platform} · ${c.model} · ${c.ip||'—'}`).join('；')||'尚无客户端';
    const online=overview.worker && Date.now()/1000-overview.worker.heartbeat<30;
    $('workerState').textContent=online?'自动处理运行中':'工作进程未就绪';
    $('workerDetail').textContent=online?(overview.worker.message.startsWith('处理中')?'正在分析视频':'等待新任务'):'API 已连接，请检查 worker';
    $('connection').textContent='● 服务已连接';
    if(state.task){
      const task=await api('/v1/tasks/'+state.task.task_id);
      if(task.revision!==state.task.revision){if(state.dirty)$('changed').hidden=false;else await selectTask(task.task_id);}
    }else if(state.records.length){await selectTask(state.records[0].task_id);}
  }catch(error){$('connection').textContent='服务连接中断';if(!$('loginDialog').open)toast(error.message);}finally{state.loading=false;}
}
function renderRecords(){
  $('recordList').replaceChildren();
  if(!state.records.length)$('recordList').append(element('p','暂无匹配记录。导入视频或从手机标记一段片段。','empty-list'));
  for(const task of state.records){
    const item=element('button',undefined,'record'+(state.task?.task_id===task.task_id?' active':''));
    const top=element('div',undefined,'record-top');top.append(element('strong',task.effective_result?.plate||'车牌未确认'),badge(task.status));
    item.append(top,element('div',`${title(task.metadata.trigger||'import')} · ${title(task.effective_result?.violation_type||'UNKNOWN')}`,'sub'),element('small',date(task.created_at)));
    if(task.review)item.append(element('small',title(task.review.decision)));
    if(task.submission_status==='SUBMITTED')item.append(element('small','已提交 · 判定锁定'));
    item.onclick=run(async()=>{if(state.dirty&&!confirm('未保存的干预内容将丢弃，继续切换记录？'))return;state.dirty=false;await selectTask(task.task_id);});
    $('recordList').append(item);
  }
  $('page').textContent=String(state.offset/20+1);$('previous').disabled=state.offset===0;$('next').disabled=state.records.length<20;
}
async function selectTask(id){
  const [task,audit]=await Promise.all([api('/v1/tasks/'+id),api('/v1/tasks/'+id+'/audit')]);
  const changedVideo=state.task?.task_id!==id||!!$('video').error;state.task=task;state.dirty=false;state.scene=structuredClone(task.scene||{});state.draw=null;state.points=[];
  $('empty').hidden=true;$('selected').hidden=false;$('changed').hidden=true;
  $('taskTime').textContent=date(task.created_at);$('taskTitle').textContent=title(task.metadata.trigger||'import')+' · '+(task.effective_result?.plate||'车牌未确认');
  $('taskId').textContent=task.event_id;$('taskStatus').replaceWith(Object.assign(badge(task.status),{id:'taskStatus'}));
  const ai=task.result||{},effective=task.effective_result||{};
  fields($('aiResult'),[['车牌',ai.plate||'未确认'],['信号灯',title(ai.signal_state?.color)+(ai.signal_state?.stable?'（稳定）':'')],['红灯观察',ai.signal_approach?ai.signal_approach.reason:'尚无足够红灯采样'],['违法行为',title(ai.violation_type||'UNKNOWN')],['判断',title(ai.decision)],['原因',title(ai.reason)],['车辆模型',ai.model?.name||task.analysis_config?.vehicle_model||'—'],['车牌模型',ai.plate_model||'未启用 / 旧结果']]);
  fields($('effectiveResult'),[['来源',effective.source==='HUMAN'?'人工干预':'自动判断'],['车牌',effective.plate||'未确认'],['违法行为',title(effective.violation_type||'UNKNOWN')],['状态',title(effective.review_status==='AUTOMATIC'?effective.decision:effective.review_status)],['提交',task.submission_status==='SUBMITTED'?'已提交，判定锁定':'尚未提交 · 可人工复核']]);
  $('rawResult').textContent=JSON.stringify(ai,null,2);
  $('correctPlate').value=effective.plate||'';$('correctViolation').value=effective.violation_type||'UNKNOWN';$('reviewDecision').value=task.review?.decision||'VALID';$('reviewNote').value=task.review?.note||'';
  const editable=task.submission_status==='NOT_SUBMITTED'&&['ANALYZED','REJECTED','ERROR'].includes(task.status);
  $('reviewFields').disabled=!editable;$('reviewLock').textContent=editable?'自动结果已经生效，必要时在此纠正。':(task.submission_status==='SUBMITTED'?'已提交记录只读。':'分析完成后可进行人工干预。');
  $('reanalyze').disabled=!editable||task.status==='EXPIRED';$('fixedCamera').checked=!!state.scene.fixed_camera;
  $('videoError').hidden=true;$('download').href=`/v1/tasks/${id}/video?original=true`;
  if(changedVideo){mediaRetry=false;state.clipStart=0;$('video').src=`/v1/tasks/${id}/video`;$('video').load();}
  $('videoInfo').textContent=ai.video?`${ai.video.width}×${ai.video.height} · ${ai.video.duration_seconds.toFixed(1)} 秒 · ${ai.sampled_frames} 采样帧`:title(task.status);
  $('plateEvidence').replaceChildren();
  const plates=[...(ai.plates||[])].sort((a,b)=>plateTrust(b)-plateTrust(a)||(b.hits||0)-(a.hits||0));
  plates.forEach((plate,index)=>{
    const t=plates.length===1?1:1-index/(plates.length-1);
    const tone=plateHeat(t);
    const b=element('button',undefined,'plate-card ranked');
    b.style.setProperty('--plate-bg',tone.bg);b.style.setProperty('--plate-fg',tone.fg);
    b.append(element('strong',plate.text),element('small',`${plate.stable?'多帧一致':'待确认'} · ${plate.hits} 帧 · ${Math.round(plate.confidence*100)}%`));
    b.onclick=()=>{const video=$('video'),src=`/v1/tasks/${id}/video`;state.clipStart=0;video.pause();if(video.getAttribute('src')!==src){video.src=src;video.load();}video.currentTime=plate.evidence_time;draw();};
    $('plateEvidence').append(b);
  });
  if(!plates.length)$('plateEvidence').append(element('p','尚无满足阈值的中文车牌。小尺寸、模糊或非中国大陆车牌可能无法识别。','muted'));
  $('violations').replaceChildren();
  for(const violation of ai.violations||[]){const card=element('div',undefined,'candidate');card.append(element('strong',title(violation.type)),element('p',`${violation.plate||'车牌未关联'} · 轨迹${violation.track_id??'—'} · ${violation.reason}`));const b=element('button',violation.clip_index!=null?`看裁剪 ${Number(violation.clip_start).toFixed(1)}–${Number(violation.clip_end).toFixed(1)} 秒`:`定位 ${violation.time_seconds.toFixed(1)} 秒`);b.onclick=()=>{if(violation.clip_index!=null){state.clipStart=violation.clip_start||0;$('video').src=`/v1/tasks/${id}/clips/${violation.clip_index}`;$('video').load();$('video').play();}else{$('video').currentTime=violation.time_seconds;$('video').pause();}};card.append(b);$('violations').append(card);}
  if(!ai.violations?.length)$('violations').append(element('p',title(ai.rule_assessment||ai.reason||'UNKNOWN'),'muted'));
  $('audit').replaceChildren();for(const row of audit.history){const payload=row.payload,entry=element('div',undefined,'audit-item');entry.append(element('strong',row.kind==='REVIEW'?title(payload.decision):(row.kind==='REANALYZE'?'按新参数重新分析':'已记录提交凭证')),element('small',date(row.created_at)+(payload.reviewer?' · '+payload.reviewer:'')));if(payload.note)entry.append(element('p',payload.note));$('audit').append(entry);}
  if(!audit.history.length)$('audit').append(element('p','暂无人工修改。该记录由系统自动处理。','muted'));
  calibrationHint();renderRecords();draw();
}
function layout(){const video=$('video'),canvas=$('overlay');canvas.width=video.clientWidth;canvas.height=video.clientHeight;const vw=video.videoWidth||16,vh=video.videoHeight||9;const scale=Math.min(canvas.width/vw,canvas.height/vh);return {w:vw*scale,h:vh*scale,x:(canvas.width-vw*scale)/2,y:(canvas.height-vh*scale)/2};}
function draw(){
  const canvas=$('overlay'),box=layout(),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);if(!state.task)return;
  const frames=state.task.result?.frames||[],now=$('video').currentTime+(state.clipStart||0);
  const frame=frames.reduce((best,f)=>!best||Math.abs(f.time_seconds-now)<Math.abs(best.time_seconds-now)?f:best,null);
  if($('showBoxes').checked&&frame&&Math.abs(frame.time_seconds-now)<=1/(state.task.result.sample_fps||2)){
    const lamp={RED:'#ff6b6b',GREEN:'#6bff9a',YELLOW:'#ffe184'};
    for(const detection of [...frame.vehicles.map(v=>({...v,text:title(v.label),color:'#76d5cf'})),...(frame.plates||[]).map(p=>({...p,color:'#ffe184'})),...(frame.lights||[]).map(p=>({...p,text:title(p.color),color:lamp[p.color]||'#ffe184'}))]){
      const [x1,y1,x2,y2]=detection.box_normalized;ctx.strokeStyle=detection.color;ctx.lineWidth=1.5;ctx.strokeRect(box.x+x1*box.w,box.y+y1*box.h,(x2-x1)*box.w,(y2-y1)*box.h);ctx.font='12px sans-serif';const text=detection.text,labelWidth=ctx.measureText(text).width+9;const tx=Math.max(0,Math.min(box.x+x1*box.w,canvas.width-labelWidth)),ty=Math.max(16,box.y+y1*box.h);ctx.fillStyle='#10232de0';ctx.fillRect(tx,ty-15,labelWidth,16);ctx.fillStyle=detection.color;ctx.fillText(text,tx+4,ty-3);
    }
  }
  for(const [key,color] of [['solid_line','#ffcc66'],['allowed_direction','#7cd8a9']]){const line=state.scene[key];if(!line)continue;ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(box.x+line[0][0]*box.w,box.y+line[0][1]*box.h);ctx.lineTo(box.x+line[1][0]*box.w,box.y+line[1][1]*box.h);ctx.stroke();ctx.fillStyle=color;ctx.fillText(key==='solid_line'?'实线':'允许方向 →',box.x+line[1][0]*box.w,box.y+line[1][1]*box.h-8);}
  for(const point of state.points){ctx.fillStyle='#ffcc66';ctx.beginPath();ctx.arc(box.x+point[0]*box.w,box.y+point[1]*box.h,5,0,Math.PI*2);ctx.fill();}
}
function calibrationHint(){ $('calibrationHint').textContent=state.draw?'请在视频中点击两点，标定完成后重新分析。':`实线：${state.scene.solid_line?'已标定':'未标定'} · 通行方向：${state.scene.allowed_direction?'已标定':'未标定'}`;document.querySelector('.video-wrap').classList.toggle('calibrating',!!state.draw); }
$('overlay').onclick=event=>{if(!state.draw)return;const rect=$('overlay').getBoundingClientRect(),box=layout();const point=[(event.clientX-rect.left-box.x)/box.w,(event.clientY-rect.top-box.y)/box.h];if(point.some(x=>x<0||x>1))return;state.points.push(point.map(x=>Math.round(x*10000)/10000));if(state.points.length===2){state.scene[state.draw]=state.points;state.points=[];state.draw=null;state.dirty=true;calibrationHint();}draw();};
$('video').addEventListener('timeupdate',draw);$('video').addEventListener('loadedmetadata',draw);new ResizeObserver(draw).observe($('video'));$('showBoxes').onchange=draw;
$('video').addEventListener('loadeddata',()=>{$('videoError').hidden=true;});
$('video').onerror=async()=>{
  if(!mediaRetry){
    mediaRetry=true;session='';
    try{await ensureSession();$('video').load();return;}catch{}
  }
  $('videoError').textContent='视频预览暂不可用，可能正在生成或视频已过期。可刷新任务或下载原片。';$('videoError').hidden=false;
};
$('drawLine').onclick=()=>{state.draw='solid_line';state.points=[];$('video').pause();calibrationHint();};$('drawDirection').onclick=()=>{state.draw='allowed_direction';state.points=[];$('video').pause();calibrationHint();};
$('clearScene').onclick=()=>{state.scene={};state.draw=null;state.points=[];state.dirty=true;$('fixedCamera').checked=false;calibrationHint();draw();};$('fixedCamera').onchange=()=>{state.scene.fixed_camera=$('fixedCamera').checked;state.dirty=true;};
$('reviewForm').oninput=()=>state.dirty=true;
$('reviewForm').onsubmit=run(async()=>{const task=state.task;const body={expected_revision:task.revision,decision:$('reviewDecision').value,plate:$('correctPlate').value.trim().toUpperCase(),violation_type:$('correctViolation').value,reviewer:$('reviewer').value.trim(),note:$('reviewNote').value.trim()};await api(`/v1/tasks/${task.task_id}/review`,{method:'POST',json:body});state.dirty=false;await selectTask(task.task_id);toast('人工结果已保存，手机会自动同步。');await refresh();});
$('reanalyze').onclick=run(async()=>{if(!state.task)return;const settings=await api('/v1/settings');await api(`/v1/tasks/${state.task.task_id}/reanalyze`,{method:'POST',json:{expected_revision:state.task.revision,config:settings.config,scene:{...state.scene,fixed_camera:$('fixedCamera').checked}}});state.dirty=false;await selectTask(state.task.task_id);toast('已重新排队，将自动分析并回传。');});
$('reloadTask').onclick=run(async()=>{state.dirty=false;await selectTask(state.task.task_id);});
$('archive').onclick=run(async()=>{if(!confirm('归档将清除当前全部记录和视频，且不可恢复。继续？'))return;const out=await api('/v1/archive',{method:'POST'});state.task=null;state.records=[];state.clipStart=0;$('selected').hidden=true;$('empty').hidden=false;$('video').removeAttribute('src');$('video').load();await refresh();toast(`已归档 ${out.archived} 条记录`);});
$('pair').onclick=run(async()=>{const out=await api('/v1/device-pair',{method:'POST'});toast(`在手机「访问令牌」填入：${out.code}（30分钟有效）`);});
$('refresh').onclick=run(refresh);for(const id of ['statusFilter','reviewFilter'])$(id).onchange=()=>{state.offset=0;refresh();};$('search').oninput=()=>{clearTimeout(state.searchTimer);state.searchTimer=setTimeout(()=>{state.offset=0;refresh();},300);};
$('previous').onclick=()=>{state.offset=Math.max(0,state.offset-20);refresh();};$('next').onclick=()=>{state.offset+=20;refresh();};
$('settingsButton').onclick=run(async()=>{state.config=await api('/v1/settings');const c=state.config.config;$('vehicleModel').replaceChildren();for(const model of state.config.models){const o=element('option',model.name+(model.installed?'':'（未下载）'));o.value=model.id;o.disabled=!model.installed;$('vehicleModel').append(o);}$('vehicleModel').value=c.vehicle_model;$('plateModel').replaceChildren();for(const model of state.config.plate_models||[]){const o=element('option',model.name+(model.installed?'':'（未下载）'));o.value=model.id;o.disabled=!model.installed;$('plateModel').append(o);}$('plateModel').value=c.plate_model||'hyperlpr3';for(const [id,key] of [['vehicleThreshold','vehicle_threshold'],['sampleFps','sample_fps'],['plateThreshold','plate_threshold'],['plateMinHits','plate_min_hits'],['threads','threads']])$(id).value=c[key];$('plateEnabled').checked=c.plate_enabled;$('rulesEnabled').checked=c.rules_enabled;$('modelReadiness').textContent=(state.config.plate_models||[]).filter(m=>m.installed).map(m=>m.name).join('；')||'车牌模型尚未下载';$('settingsDialog').showModal();});
$('settingsForm').onsubmit=run(async()=>{const config={vehicle_model:$('vehicleModel').value,plate_model:$('plateModel').value,vehicle_threshold:+$('vehicleThreshold').value,sample_fps:+$('sampleFps').value,plate_threshold:+$('plateThreshold').value,plate_min_hits:+$('plateMinHits').value,threads:+$('threads').value,plate_enabled:$('plateEnabled').checked,rules_enabled:$('rulesEnabled').checked};await api('/v1/settings',{method:'PUT',json:{expected_revision:state.config.revision,config}});$('settingsDialog').close();toast('参数已保存，对之后的新任务生效。');});$('closeSettings').onclick=()=>$('settingsDialog').close();
$('loginDialog').addEventListener('cancel',event=>event.preventDefault());
$('loginForm').onsubmit=async event=>{event.preventDefault();$('loginError').textContent='';try{await api('/v1/session',{method:'POST',headers:{Authorization:'Bearer '+$('token').value.trim()}});$('token').value='';$('loginDialog').close();await refresh();}catch(error){$('loginError').textContent=error.message;}};
$('logout').onclick=run(async()=>{await api('/v1/session',{method:'DELETE'});state.task=null;state.records=[];$('selected').hidden=true;$('empty').hidden=false;$('video').removeAttribute('src');$('video').load();$('recordList').replaceChildren();$('loginDialog').showModal();});
$('importButton').onclick=()=>$('videoFile').click();$('videoFile').onchange=run(async()=>{const file=$('videoFile').files[0];if(!file)return;if(file.size>50*1024*1024)throw new Error('视频超过 50 MiB');$('importButton').disabled=true;try{const body=await file.arrayBuffer();const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',body)),n=>n.toString(16).padStart(2,'0')).join('');const task=await api('/v1/tasks',{method:'POST',body,headers:{'Content-Type':'video/mp4','X-Video-SHA256':hash,'X-Event-Metadata':JSON.stringify({event_id:crypto.randomUUID(),trigger:'import',manual_review:false})}});state.offset=0;await selectTask(task.task_id);await refresh();toast('视频已上传，正在自动处理。');}finally{$('importButton').disabled=false;$('videoFile').value='';}});
refresh();setInterval(refresh,4000);
$('logout').hidden=true;$('pair').hidden=true;
