"""Conservative geometric candidates for a calibrated, stationary camera; never legal proof."""
from collections import Counter

import cv2
import numpy as np


class MotionGate:
    def __init__(self):
        self.previous=None
        self.moving=0
        self.valid=0

    def update(self,image,boxes):
        gray=cv2.cvtColor(cv2.resize(image,(320,180)),cv2.COLOR_BGR2GRAY)
        mask=np.full(gray.shape,255,np.uint8)
        for x1,y1,x2,y2 in boxes:
            cv2.rectangle(mask,(int(x1*320),int(y1*180)),(int(x2*320),int(y2*180)),0,-1)
        if self.previous is not None:
            points=cv2.goodFeaturesToTrack(self.previous,150,.02,8,mask=mask)
            if points is not None and len(points)>=20:
                next_points,ok,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,points,None)
                if next_points is not None and ok.sum()>=15:
                    shifts=next_points[ok[:,0]==1]-points[ok[:,0]==1]
                    self.valid+=1
                    if float(np.median(np.linalg.norm(shifts,axis=2)))>2.0:
                        self.moving+=1
        self.previous=gray

    def status(self):
        if self.valid<2: return 'INSUFFICIENT_BACKGROUND'
        return 'MOVING_CAMERA' if self.moving else 'STATIONARY'


def candidates(frames,scene,motion_status,enabled=True):
    if not enabled: return [],'DISABLED'
    if not scene or not scene.get('fixed_camera') or not (scene.get('solid_line') or scene.get('allowed_direction')):
        return [],'NEEDS_CALIBRATION'
    if motion_status!='STATIONARY': return [],motion_status
    roi=scene.get('road_roi',[0,0,1,1])
    tracks={}
    for frame in frames:
        for v in frame['vehicles']:
            x1,y1,x2,y2=v['box_normalized']; point=np.array([(x1+x2)/2,y2])
            if roi[0]<=point[0]<=roi[2] and roi[1]<=point[1]<=roi[3]:
                tracks.setdefault(v['track_id'],[]).append((frame['time_seconds'],point,v['score']))
    result=[]
    for track,points in tracks.items():
        if len(points)<3 or points[-1][0]-points[0][0]<1: continue
        line=scene.get('solid_line')
        if line:
            a,b=np.array(line); direction=b-a; length=np.linalg.norm(direction)
            distances=[float((direction[0]*(p[1]-a[1])-direction[1]*(p[0]-a[0]))/length) for _,p,_ in points]
            for i in range(1,len(points)):
                if distances[i-1]*distances[i]<0 and abs(distances[i-1])+abs(distances[i])>.025:
                    p=points[i][1]; projection=float(np.dot(p-a,direction)/(length*length))
                    if 0<=projection<=1:
                        result.append({'type':'SOLID_LINE','track_id':track,'time_seconds':points[i][0],
                            'confidence':round(min(p[2] for p in points),4),'reason':'轨迹底边中心跨越已标定实线，需结合车轮位置复核'})
                        break
        allowed=scene.get('allowed_direction')
        if allowed:
            a,b=np.array(allowed); direction=(b-a)/np.linalg.norm(b-a)
            movement=points[-1][1]-points[0][1]
            steps=np.array([np.dot(points[i][1]-points[i-1][1],direction) for i in range(1,len(points))])
            if np.dot(movement,direction)<-.08 and (steps<-.003).mean()>=.75:
                result.append({'type':'WRONG_WAY','track_id':track,'time_seconds':points[-1][0],
                    'confidence':round(min(p[2] for p in points),4),'reason':'连续轨迹与已标定通行方向相反'})
    return result,'ASSESSED'


def _usable(vehicle):
    x1,y1,x2,y2=vehicle['box_normalized']
    cx,area=(x1+x2)/2,(x2-x1)*(y2-y1)
    # Hood / dashboard boxes sit on the bottom edge and drown the car ahead.
    if y2<.35 or y2>.94 or y1>.72 or not .008<=area<=.35:
        return False
    return .05<=cx<=.95


def _box_point(vehicle, time_seconds):
    x1,y1,x2,y2=vehicle['box_normalized']
    return (time_seconds,y2,(x2-x1)*(y2-y1),vehicle.get('score',0),(x1+x2)/2)


def _points_for_track(window, track_id):
    points=[]
    for frame in window:
        for vehicle in frame.get('vehicles') or []:
            if vehicle.get('track_id')==track_id:
                points.append(_box_point(vehicle, frame['time_seconds']))
                break
    return points


def _peer_median(window, start=None, end=None):
    hits={}
    for frame in window:
        t=frame['time_seconds']
        if start is not None and (t<start or t>end):
            continue
        for vehicle in frame.get('vehicles') or []:
            if _usable(vehicle):
                hits.setdefault(vehicle['track_id'],[]).append(_box_point(vehicle, t))
    deltas=[pts[-1][1]-pts[0][1] for pts in hits.values() if len(pts)>=3]
    scores={tid: (pts[-1][4]-pts[0][4], pts[-1][1]-pts[0][1])
            for tid,pts in hits.items() if len(pts)>=3}
    median=sorted(deltas)[len(deltas)//2] if len(deltas)>=2 else None
    return median, scores


def _vehicle_for_plate(frame, plate_text):
    vehicles=frame.get('vehicles') or []
    for plate in frame.get('plates') or []:
        if plate.get('text')!=plate_text:
            continue
        tid=plate.get('track_id')
        if tid is not None:
            for vehicle in vehicles:
                if vehicle['track_id']==tid:
                    return vehicle
        box=plate.get('box_normalized')
        if box and len(box)==4:
            cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
            containing=[v for v in vehicles
                        if v['box_normalized'][0]<=cx<=v['box_normalized'][2]
                        and v['box_normalized'][1]<=cy<=v['box_normalized'][3]]
            if containing:
                return min(containing,key=lambda v:(
                    v['box_normalized'][2]-v['box_normalized'][0])*
                    (v['box_normalized'][3]-v['box_normalized'][1]))
    return None


def red_approach(frames, plate_text=None):
    """红灯期间跟同一条轨迹，不把超车位移记到等灯车上。移动机位不是越线。"""
    red=[f for f in frames if f.get('signal_observed')=='RED']
    windows,current=[],[]
    for frame in red:
        if current and frame['time_seconds']-current[-1]['time_seconds']>1.0:
            windows.append(current)
            current=[]
        current.append(frame)
    if current:
        windows.append(current)

    def summarize(points,track_id,peer_med=None,scores=None):
        delta=points[-1][1]-points[0][1]
        dcx=points[-1][4]-points[0][4]
        area0,area1=points[0][2],points[-1][2]
        approaching=delta>.03
        receding=delta<-.03
        lateral=abs(dcx)>.12
        area_jump=abs(area1-area0)>.35*max(area0,.01) and abs(delta)>.015
        proceeding=approaching or receding or area_jump or lateral
        if peer_med is not None and not lateral and not receding:
            unique=abs(delta-peer_med)>.04
            proceeding=approaching and (unique or abs(delta)>.08)
        mine=abs(dcx)+abs(delta)
        magnitudes=[abs(dx)+abs(dy) for dx,dy in (scores or {}).values()]
        best=max([mine, *magnitudes])
        if best>.06 and mine<.6*best:
            proceeding=False
        others=[pair for tid,pair in (scores or {}).items() if tid!=track_id]
        if others and not receding and not lateral:
            if any(dy<-.03 or abs(dx)>.12 for dx,dy in others):
                proceeding=False
        return {
            'track_id':track_id,'frames':len(points),
            'first_bottom':round(float(points[0][1]),4),'last_bottom':round(float(points[-1][1]),4),
            'delta_bottom':round(float(delta),4),'approaching':approaching,'proceeding':proceeding,
            'time_seconds':round(float(points[-1][0]),3),
            'reason':('红灯期间前车仍在画面中移动；移动机位不能据此判定越线'
                      if proceeding else '红灯期间前车底边未明显变化'),
        }

    def best_local(points, track_id, window):
        found=[]
        last_t=-99
        for i, start in enumerate(points):
            if start[0]<last_t+2:
                continue
            local=[p for p in points[i:] if p[0]-start[0]<=4]
            if len(local)<4:
                continue
            peer,scores=_peer_median(window, local[0][0], local[-1][0])
            item=summarize(local, track_id, peer, scores)
            if item['proceeding']:
                found.append(item)
                last_t=item['time_seconds']
        if found:
            return max(found,key=lambda item:(item['proceeding'],abs(item['delta_bottom']),item['frames']))
        return summarize(points, track_id, *_peer_median(window)) if len(points)>=3 else None

    found=[]
    for window in windows:
        if len(window)<4 or window[-1]['time_seconds']-window[0]['time_seconds']<1:
            continue
        if plate_text:
            votes=Counter()
            for frame in window:
                vehicle=_vehicle_for_plate(frame, plate_text)
                if vehicle:
                    votes[vehicle['track_id']]+=1
            if not votes:
                continue
            def plate_key(tid):
                pts=_points_for_track(window, tid)
                span=abs(pts[-1][1]-pts[0][1]) if len(pts)>=3 else -1
                return (votes[tid], span, len(pts))
            track_id=max(votes, key=plate_key)
            points=_points_for_track(window, track_id)
            item=best_local(points, track_id, window)
            if item:
                found.append(item)
            continue
        hits={}
        for frame in window:
            for vehicle in frame.get('vehicles') or []:
                if _usable(vehicle):
                    hits.setdefault(vehicle['track_id'],[]).append(
                        _box_point(vehicle, frame['time_seconds']))
        if not hits:
            continue
        def span(pts):
            return abs(pts[-1][1]-pts[0][1]) if len(pts)>=3 else -1
        track_id,points=max(hits.items(),key=lambda item:(span(item[1]),len(item[1])))
        item=best_local(points, track_id, window)
        if item:
            found.append(item)
    if not found:
        return None
    return max(found,key=lambda item:(item['proceeding'],abs(item['delta_bottom']),item['frames']))


def red_approaches(frames, plate_texts=None):
    """每位稳定车牌一条接近观察，供同一红灯下多车候选。"""
    texts=[text for text in (plate_texts or []) if text]
    if not texts:
        item=red_approach(frames)
        return [item] if item else []
    found,seen=[],set()
    for text in texts:
        item=red_approach(frames, text)
        if not item or item['track_id'] in seen:
            continue
        found.append({**item,'plate':text})
        seen.add(item['track_id'])
    return found


def red_light_candidate(signal_state, approach, plates, plate_text=None):
    """三条件同时成立才给候选：稳定红灯、红灯期间继续接近、车牌多帧一致。不是法律证明。"""
    if not signal_state or signal_state.get('color')!='RED' or not signal_state.get('stable'):
        return None
    if not approach or not (approach.get('proceeding') or approach.get('approaching')):
        return None
    plate=plate_text or approach.get('plate') or next((p['text'] for p in plates if p.get('stable') and p.get('text')), None)
    if not plate:
        return None
    return {
        'type':'RED_LIGHT','track_id':approach['track_id'],
        'time_seconds':approach.get('time_seconds') or signal_state.get('since_seconds') or 0,
        'confidence':round(min(.8, .4+approach['frames']*.03),4),
        'plate':plate,
        'reason':'红灯稳定、红灯期间车辆仍在移动、车牌多帧一致；移动机位无法确认停止线，仅作候选',
    }


def lane_changes(frames):
    """横向明显变道。2fps 看不到转向灯，也没有实线分割，只作复核候选。"""
    tracks={}
    for frame in frames:
        for vehicle in frame.get('vehicles') or []:
            x1,y1,x2,y2=vehicle['box_normalized']
            cx,area=(x1+x2)/2,(x2-x1)*(y2-y1)
            if y2>.96 or y1>.80 or not .008<=area<=.22 or not .05<=cx<=.95:
                continue
            tracks.setdefault(vehicle['track_id'],[]).append(
                (frame['time_seconds'],cx,y2,area,vehicle.get('score',0)))
    result=[]
    for track_id,points in tracks.items():
        last_t=-99
        for i, start in enumerate(points):
            if start[0]<last_t+8:
                continue
            window=[item for item in points[i:] if item[0]-start[0]<=4]
            if len(window)<4:
                continue
            delta=window[-1][1]-window[0][1]
            if abs(delta)<.08:
                continue
            t=next((item[0] for item in window if abs(item[1]-window[0][1])>=.08), window[-1][0])
            result.append({
                'type':'LATERAL_MOVEMENT','track_id':track_id,'time_seconds':round(float(t),3),
                'confidence':round(min(.55,.3+abs(delta)),4),
                'reason':'横向明显变道；未标定实线、2fps 无法确认转向灯，需复核是否压实线或不打灯变道',
            })
            last_t=t
    return result


def turned_from_green(frames, track_id, time_seconds, span=4):
    """左转/变道前本向绿灯，转过去才看到红灯：同一辆车的连续行为，不当闯红灯。"""
    colors=[]
    for frame in frames:
        if not time_seconds-span <= frame['time_seconds'] < time_seconds:
            continue
        if any(v.get('track_id')==track_id for v in frame.get('vehicles') or []):
            colors.append(frame.get('signal_observed'))
    return bool(colors) and colors.count('GREEN')>=max(1, colors.count('RED'))


def red_during_laterals(frames, laterals, seen=None):
    """同一红灯下，每条变道轨迹一张闯红灯候选；左转绿转红不报；不覆盖已有红灯卡。"""
    taken=set(seen or ())
    extra=[]
    for item in laterals:
        if item['track_id'] in taken:
            continue
        if turned_from_green(frames, item['track_id'], item['time_seconds']):
            continue
        nearby=[frame for frame in frames
                if abs(frame['time_seconds']-item['time_seconds'])<=2]
        if any(frame.get('signal_observed')=='RED' for frame in nearby):
            extra.append({
                'type':'RED_LIGHT','track_id':item['track_id'],
                'time_seconds':item['time_seconds'],'confidence':0.45,
                'reason':'本向红灯期间横向变道；未确认停止线与转向灯，仅作候选',
            })
            taken.add(item['track_id'])
    return extra


def restricted_park(frames):
    """右侧长时间几乎不动，疑似占用非机动车道。整段静帧不报。"""
    all_tracks, parked = {}, {}
    for frame in frames:
        for vehicle in frame.get('vehicles') or []:
            x1,y1,x2,y2=vehicle['box_normalized']
            cx,area=(x1+x2)/2,(x2-x1)*(y2-y1)
            row=(frame['time_seconds'],cx,y2,area)
            all_tracks.setdefault(vehicle['track_id'],[]).append(row)
            if cx>=.62 and .45<=y2<=.94 and .015<=area<=.12:
                parked.setdefault(vehicle['track_id'],[]).append(row)
    if not any(len(p)>=2 and (abs(p[-1][1]-p[0][1])>.03 or abs(p[-1][2]-p[0][2])>.03)
               for p in all_tracks.values()):
        return []
    result=[]
    for track_id,points in parked.items():
        if len(points)<6 or points[-1][0]-points[0][0]<2:
            continue
        dy=abs(points[-1][2]-points[0][2])
        da=abs(points[-1][3]-points[0][3])
        if dy>.04 or da>.25*max(points[0][3],.01):
            continue
        result.append({
            'type':'RESTRICTED_LANE','track_id':track_id,'time_seconds':round(float(points[-1][0]),3),
            'confidence':0.35,
            'reason':'右侧停驻时间较长，疑似占用非机动车道，需结合车道线复核',
        })
    return result
