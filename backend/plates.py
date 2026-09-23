"""Plate detection/recognition is selectable:
HyperLPR3 20230229 (Apache-2.0), PP-OCRv5 rec on HyperLPR det, or
OnnxOCR 2026 plate det+rec (Apache-2.0).
See models/plate/LICENSE.txt, PADDLEOCR-NOTICE.txt, ONNXOCR-NOTICE.txt.
"""
import math
import re
import cv2
import numpy as np
import onnxruntime as ort
from model_catalog import verify, verify_text, PLATE_FILES as FILES, PLATE_MODELS, PPOCR_KEYS, PPOCR_KEYS_SHA256
TOKENS = ['blank', "'", *'0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ',
          *'云京冀吉学宁川挂新晋桂民沪津浙渝港湘琼甘皖粤航苏蒙藏警豫贵赣辽鄂闽陕青鲁黑领使澳']
ONNXOCR_CHARS = (
    '#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新'
    '学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品'
)
FORMAT = re.compile(r'^[云京冀吉宁川新晋桂沪津浙渝湘琼甘皖粤苏蒙藏豫贵赣辽鄂闽陕青鲁黑][A-Z][A-HJ-NP-Z0-9]{4,5}[A-HJ-NP-Z0-9学挂警港澳]$')


def overlap(a, b):
    """IoU of two [x1,y1,x2,y2] boxes."""
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0


def ctc_decode(predictions, tokens):
    indexes, scores = predictions.argmax(1), predictions.max(1)
    keep = (indexes != 0) & np.r_[True, indexes[1:] != indexes[:-1]]
    if not keep.any() or int(indexes.max()) >= len(tokens):
        return '', 0.
    return ''.join(tokens[i] for i in indexes[keep]), float(scores[keep].mean())


def onnxocr_decode(logits):
    shifted = logits - logits.max(1, keepdims=True)
    exp = np.exp(shifted)
    prob = exp / np.clip(exp.sum(1, keepdims=True), 1e-9, None)
    indexes, scores = prob.argmax(1), prob.max(1)
    previous = 0
    chars, confs = [], []
    for pred, score in zip(indexes, scores):
        pred = int(pred)
        if pred != 0 and pred != previous and pred < len(ONNXOCR_CHARS):
            chars.append(ONNXOCR_CHARS[pred])
            confs.append(float(score))
        previous = pred
    return ''.join(chars), (sum(confs) / len(confs) if confs else 0.)


class PlateReader:
    def __init__(self, directory, threads=2, rec_id='hyperlpr3'):
        self.rec_id = rec_id if rec_id in PLATE_MODELS else 'hyperlpr3'
        spec = PLATE_MODELS[self.rec_id]
        self.kind = spec['kind']
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        if self.kind == 'onnxocr':
            det, rec = directory / spec['det'], directory / spec['file']
            verify(det, spec['det_sha256'])
            verify(rec, spec['sha256'])
            self._ppocr = False
        elif self.kind == 'hyperlpr':
            det, rec = directory / 'y5fu_640x_sim.onnx', directory / 'rpv3_mdict_160_r3.onnx'
            verify(det, FILES['y5fu_640x_sim.onnx'])
            verify(rec, FILES['rpv3_mdict_160_r3.onnx'])
            self.tokens = TOKENS
            self._ppocr = False
        else:
            det, rec = directory / 'y5fu_640x_sim.onnx', directory / spec['file']
            verify(det, FILES['y5fu_640x_sim.onnx'])
            verify(rec, spec['sha256'])
            keys = directory / PPOCR_KEYS
            verify_text(keys, PPOCR_KEYS_SHA256)
            self.tokens = ['blank', *keys.read_text(encoding='utf-8').splitlines(), ' ']
            self._ppocr = True
        self.detector = ort.InferenceSession(str(det), options, providers=['CPUExecutionProvider'])
        self.recognizer = ort.InferenceSession(str(rec), options, providers=['CPUExecutionProvider'])

    def text(self, crop):
        h, w = crop.shape[:2]
        if self.kind == 'onnxocr':
            resized = cv2.resize(crop, (168, 48)).astype(np.float32)
            tensor = ((resized / 255 - 0.588) / 0.193).transpose(2, 0, 1)[None]
            logits = self.recognizer.run(None, {self.recognizer.get_inputs()[0].name: tensor})[0][0]
            return onnxocr_decode(logits)
        if self._ppocr:
            ratio = w / max(h, 1)
            resized_w = 320 if math.ceil(48 * ratio) > 320 else max(1, int(math.ceil(48 * ratio)))
            resized = cv2.resize(crop, (resized_w, 48)).astype(np.float32).transpose(2, 0, 1) / 255
            tensor = np.zeros((1, 3, 48, 320), np.float32)
            tensor[0, :, :, :resized_w] = (resized - 0.5) / 0.5
            predictions = self.recognizer.run(None, {self.recognizer.get_inputs()[0].name: tensor})[0]
            if predictions.ndim == 3:
                predictions = predictions[0]
                if predictions.shape[0] == len(self.tokens) and predictions.shape[-1] != len(self.tokens):
                    predictions = predictions.T
            text, confidence = ctc_decode(predictions, self.tokens)
            return text.replace(' ', '').replace('\u3000', ''), confidence
        resized_width = min(160, max(48, math.ceil(48 * w / h)))
        tensor = np.zeros((1, 3, 48, 160), np.float32)
        tensor[0, :, :, :resized_width] = (cv2.resize(crop, (resized_width, 48)).transpose(2, 0, 1).astype(np.float32) - 127.5) / 127.5
        predictions = self.recognizer.run(None, {self.recognizer.get_inputs()[0].name: tensor})[0][0]
        return ctc_decode(predictions, self.tokens)

    def read(self, image, threshold=.85):
        """整帧一遍 + 大图 2x2 重叠分块，按车牌文字去重保留最高置信度。

        1080p 行车记录仪画面缩到 640 后前车车牌只剩约 16px，检测直接漏掉；
        分块把有效分辨率翻倍。ponytail: 每帧检测器多跑 4 次；再不够时的升级
        路径是按车辆框裁剪检测，而不是更细的金字塔分块。
        """
        h,w = image.shape[:2]
        regions = [(0,0,w,h)]
        if min(640/h,640/w) <= .5:
            tw,th = (w+1)//2+w//10, (h+1)//2+h//10
            regions += [(x1,y1,x1+tw,y1+th) for x1 in (0,w-tw) for y1 in (0,h-th)]
        # 同一块物理车牌可能被多个 pass 检出且读数不同。掉字是已确认的失败模式
        # （多字未见），所以合并时先比字符数再比置信度。
        def better(a, b):
            return (len(a['text']), a['confidence']) > (len(b['text']), b['confidence'])
        merged = []
        for region in regions:
            for plate in self.read_region(self.crop(image,region), threshold):
                plate = self.to_frame(plate,region,w,h)
                slot = next((i for i,seen in enumerate(merged)
                             if overlap(plate['box_normalized'],seen['box_normalized']) > .5), None)
                if slot is None: merged.append(plate)
                elif better(plate,merged[slot]): merged[slot] = plate
        best = {}
        for plate in merged:
            refined = self.refine(image,plate,threshold)
            if refined and better(refined,plate):
                plate = refined
            if plate['confidence'] > best.get(plate['text'],{}).get('confidence',-1):
                best[plate['text']] = plate
        return sorted(best.values(), key=lambda p: -p['confidence'])

    def refine(self, image, plate, threshold):
        """小车牌按原图放大重读：裁剪车牌邻域让检测器上采样，角点和省份字符都更准。"""
        h,w = image.shape[:2]
        x1,y1,x2,y2 = plate['box_normalized']
        bw,bh = (x2-x1)*w,(y2-y1)*h
        # 只有整帧 letterbox 后车牌确实过小才重读；足够大时原生像素 OCR 已是最优，
        # 重采样反而可能掉字（苏ED51712 → 苏ED5112 的教训）。
        if bw < 8 or bh < 4 or bw*min(1,640/h,640/w) >= 48:
            return None
        region = (int(max(0,x1*w-2*bw)),int(max(0,y1*h-5*bh)),
                  int(min(w,x2*w+2*bw)),int(min(h,y2*h+5*bh)))
        found = self.read_region(self.crop(image,region), threshold)
        if not found:
            return None
        # 取离裁剪中心最近的，避免边缘混进邻车车牌
        nearest = min(found, key=lambda p: abs(sum(p['box_normalized'][::2])-1)+abs(sum(p['box_normalized'][1::2])-1))
        return self.to_frame(nearest,region,w,h)

    @staticmethod
    def crop(image, region):
        x1,y1,x2,y2 = region
        return image[y1:y2,x1:x2]

    @staticmethod
    def to_frame(plate, region, width, height):
        x1,y1,x2,y2 = region
        bx1,by1,bx2,by2 = plate['box_normalized']
        plate['box_normalized'] = [round(float(np.clip(v,0,1)),5) for v in
            ((x1+bx1*(x2-x1))/width,(y1+by1*(y2-y1))/height,
             (x1+bx2*(x2-x1))/width,(y1+by2*(y2-y1))/height)]
        return plate

    def read_region(self, image, threshold=.85):
        h,w = image.shape[:2]
        ratio = min(640/h,640/w)
        rh,rw = int(h*ratio),int(w*ratio)
        top,left = (640-rh)//2,(640-rw)//2
        resized = np.zeros((640,640,3), np.uint8)
        resized[top:top+rh,left:left+rw] = cv2.resize(image,(rw,rh))
        tensor = np.ascontiguousarray(resized[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
        raw = self.detector.run(None,{self.detector.get_inputs()[0].name:tensor})[0][0]
        raw = raw[(raw[:,4]>.25) & np.isfinite(raw).all(1)]
        if not len(raw): return []
        classes = raw[:, 13:]
        scores = raw[:, 4] * classes.max(1)
        boxes = np.concatenate((raw[:, :2] - raw[:, 2:4] / 2, raw[:, 2:4]), axis=1)
        keep = np.asarray(cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), .25, .5)).reshape(-1)
        results = []
        for k in keep[:20]:
            points = ((raw[k, 5:13].reshape(4, 2) - [left, top]) / ratio).astype(np.float32)
            cw = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
            ch = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
            if not 16 <= cw <= w * 2 or not 8 <= ch <= h * 2: continue
            transform = cv2.getPerspectiveTransform(points, np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]))
            crop = cv2.warpPerspective(image, transform, (cw, ch), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            if ch / cw >= 1.5: crop = np.rot90(crop)
            if classes[k].argmax() == 1:
                split = max(1, int(crop.shape[0] * .4))
                a, sa = self.text(crop[:split]); b, sb = self.text(crop[split:])
                text, confidence = a + b, (sa + sb) / 2
            else:
                text, confidence = self.text(crop)
            if not math.isfinite(confidence) or confidence<threshold or not FORMAT.fullmatch(text): continue
            x,y,bw,bh=boxes[k]
            box=[(x-left)/ratio,(y-top)/ratio,(x+bw-left)/ratio,(y+bh-top)/ratio]
            normalized=[round(float(np.clip(v/d,0,1)),5) for v,d in zip(box,(w,h,w,h))]
            results.append({'text':text,'confidence':round(confidence,4),'box_normalized':normalized,
                            'detector_confidence':round(float(scores[k]),4)})
        return results


def consensus(frames, min_hits=2):
    """同一辆车的省份误读按后缀合并，票多的省份胜出。"""
    grouped={}
    for frame in frames:
        for text in {p['text'] for p in frame.get('plates',[])}:
            plate=max((p for p in frame['plates'] if p['text']==text),key=lambda p:p['confidence'])
            row=grouped.setdefault(text,{'text':text,'hits':0,'confidence':0.,'first_seen':frame['time_seconds'],
                                        'last_seen':frame['time_seconds'],'evidence_time':frame['time_seconds']})
            row['hits']+=1
            row['last_seen']=frame['time_seconds']
            if plate['confidence']>row['confidence']:
                row.update(confidence=plate['confidence'],evidence_time=frame['time_seconds'],
                           box_normalized=plate['box_normalized'])
    rows=list(grouped.values())
    used,merged=set(),[]
    for row in sorted(rows,key=lambda item:(-item['hits'],-item['confidence'])):
        if row['text'] in used:
            continue
        suffix=row['text'][1:] if len(row['text'])>1 else row['text']
        family=[item for item in rows if (item['text'][1:] if len(item['text'])>1 else item['text'])==suffix]
        winner=max(family,key=lambda item:(item['hits'],item['confidence']))
        for item in family:
            used.add(item['text'])
        merged.append({**winner,'hits':sum(item['hits'] for item in family),
                       'first_seen':min(item['first_seen'] for item in family),
                       'last_seen':max(item['last_seen'] for item in family)})
    return sorted(({**row,'stable':row['hits']>=min_hits} for row in merged),
                  key=lambda p:(p['stable'],p['hits'],p['confidence']),reverse=True)
