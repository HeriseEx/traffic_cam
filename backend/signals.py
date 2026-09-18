"""Traffic-light color from detector boxes plus night glow spots, then a hold-based state machine.

YOLOX class 9 boxes over *this* lane beat HSV glow. Glow still fills gaps but ignores
yellow sodium lamps and off-axis street/building lights.
ponytail: a dedicated traffic-light model if housing/arrow lamps start beating this.
"""
import cv2
import numpy as np

HOLD = 3  # 2 fps → 1.5 s of the same color before the state may change


def color_of(image, box):
    """Dominant lens color inside a detector box. Works in day or night."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = int(max(0, min(x1, x2))), int(min(w, max(x1, x2)))
    y1, y2 = int(max(0, min(y1, y2))), int(min(h, max(y1, y2)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    mx, my = int((x2 - x1) * .15), int((y2 - y1) * .15)
    crop = image[y1 + my:y2 - my or y2, x1 + mx:x2 - mx or x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    glow = value >= max(80, float(np.percentile(value, 70)))
    if int(glow.sum()) < 6:
        return None
    pixels = hsv[glow]
    sat = pixels[:, 1] > 50
    if int(sat.sum()) < 4:
        return None
    hue = pixels[sat, 0]
    counts = {
        'RED': int(((hue <= 12) | (hue >= 165)).sum()),
        'GREEN': int(((hue >= 40) & (hue <= 100)).sum()),
        'YELLOW': int(((hue >= 16) & (hue <= 34)).sum()),
    }
    # LED red often blooms a yellow/white core; do not call that yellow.
    if counts['RED'] >= 3 and counts['RED'] + counts['YELLOW'] >= counts['GREEN']:
        return 'RED'
    color, n = max(counts.items(), key=lambda item: item[1])
    return color if n >= 4 else None


def _glow(image):
    h, w = image.shape[:2]
    if h < 32 or w < 32:
        return []
    y0, y1 = int(h * .08), int(h * .48)
    x0, x1 = int(w * .12), int(w * .95)
    roi = image[y0:y1, x0:x1]
    if cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean() > 70:
        return []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)
    lights = []
    # No yellow glow: sodium street lamps and windows read as yellow.
    for color, mask in (
        ('RED', cv2.inRange(hsv, (0, 80, 130), (12, 255, 255)) |
                cv2.inRange(hsv, (165, 80, 130), (180, 255, 255))),
        ('GREEN', cv2.inRange(hsv, (40, 80, 120), (100, 255, 255))),
    ):
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(closed, 8)
        for i in range(1, count):
            x, y, bw, bh, area = stats[i]
            if not 8 <= area <= 900 or max(bw, bh) > 80:
                continue
            aspect = bw / max(bh, 1)
            if not .2 <= aspect <= 3.0:
                continue
            lights.append({
                'color': color, 'area': int(area), 'score': float(area), 'source': 'HSV_GLOW',
                'box_normalized': [round(float(v), 5) for v in
                    ((x0 + x) / w, (y0 + y) / h, (x0 + x + bw) / w, (y0 + y + bh) / h)],
                'center': [round(float((x0 + centroids[i][0]) / w), 5),
                           round(float((y0 + centroids[i][1]) / h), 5)],
            })
    return lights


def detect_lights(image, boxes=None):
    h, w = image.shape[:2]
    lights = []
    for box in boxes or []:
        color = color_of(image, box)
        if not color:
            continue
        x1, y1, x2, y2 = box
        lights.append({
            'color': color, 'area': int(max(1, (x2 - x1) * (y2 - y1))),
            'score': 10000.0, 'source': 'BOX',
            'box_normalized': [round(float(v), 5) for v in
                (x1 / w, y1 / h, x2 / w, y2 / h)],
            'center': [round(float((x1 + x2) / 2 / w), 5),
                       round(float((y1 + y2) / 2 / h), 5)],
        })
    lights.extend(_glow(image))
    lights.sort(key=lambda item: -item.get('score', item['area']))
    kept = []
    for item in lights:
        if any(abs(item['center'][0] - other['center'][0]) < .03 and
               abs(item['center'][1] - other['center'][1]) < .04 for other in kept):
            continue
        kept.append(item)
        if len(kept) == 8:
            break
    return kept


def observe(lights):
    """Color of the lamp over *this* lane, not a cross-street or a window."""
    if not lights:
        return 'OFF'
    usable = [item for item in lights if item['color'] in ('RED', 'GREEN')]
    boxes = [item for item in usable if item.get('source') == 'BOX' and
             .22 <= item['center'][0] <= .78 and item['center'][1] < .55]
    pool = boxes or [item for item in usable if
                     .32 <= item['center'][0] <= .68 and item['center'][1] < .46]
    if not pool:
        return 'OFF'
    chosen = min(pool, key=lambda item: abs(item['center'][0] - .5) * 2 + abs(item['center'][1] - .38))
    return chosen['color']


class ThroughLamp:
    """Follow this-lane housing across frames. Far-right red is only acquired while inbound
    (x≤0.85, high); a lone x≈0.91 building/other-road lamp stays OFF. Coast keeps the last
    color through YOLOX gaps so a side-street green cannot steal the lock.
    ponytail: dedicated traffic-light tracker if housings still swap lanes.
    """

    def __init__(self, coast=6):
        self.x = None
        self.color = None
        self.miss = 0
        self.coast = coast

    def update(self, lights):
        boxes = [item for item in lights if item.get('source') == 'BOX'
                 and item['color'] in ('RED', 'GREEN') and item['center'][1] < .52]
        if self.x is not None and boxes:
            near = min(boxes, key=lambda item: abs(item['center'][0] - self.x) + abs(item['center'][1] - .38) * .3)
            if abs(near['center'][0] - self.x) < .14:
                self.x, self.color, self.miss = near['center'][0], near['color'], 0
                return self.color
        if self.color and self.miss < self.coast:
            self.miss += 1
            return self.color
        band = [item for item in boxes if .28 <= item['center'][0] <= .72]
        if band:
            chosen = min(band, key=lambda item: abs(item['center'][0] - .5) * 2 + abs(item['center'][1] - .38))
            self.x, self.color, self.miss = chosen['center'][0], chosen['color'], 0
            return chosen['color']
        inbound = [item for item in boxes if item['color'] == 'RED'
                   and .7 < item['center'][0] <= .85 and item['center'][1] < .42]
        if inbound and self.x is None:
            chosen = min(inbound, key=lambda item: item['center'][0])
            self.x, self.color, self.miss = chosen['center'][0], chosen['color'], 0
            return chosen['color']
        glow = observe(lights)
        if glow != 'OFF':
            self.x, self.color, self.miss = .5, glow, 0
        else:
            self.x, self.color, self.miss = None, None, 0
        return glow


class SignalMachine:
    def __init__(self, hold=HOLD):
        self.hold = hold
        self.color = 'UNKNOWN'
        self.pending = None
        self.streak = 0
        self.pending_start = None
        self.since = None
        self.last_stable = None
        self.last_stable_since = None

    def update(self, observed, time_seconds):
        if observed == self.pending:
            self.streak += 1
        else:
            self.pending, self.streak, self.pending_start = observed, 1, time_seconds
        if self.streak >= self.hold and self.pending != self.color:
            self.color, self.since = self.pending, self.pending_start
            if self.color in ('RED', 'GREEN', 'YELLOW'):
                self.last_stable, self.last_stable_since = self.color, self.since
        return self.snapshot(observed)

    def snapshot(self, observed=None):
        display = self.color if self.color in ('RED', 'GREEN', 'YELLOW') else (self.last_stable or self.color)
        return {
            'color': display,
            'stable': display in ('RED', 'GREEN', 'YELLOW'),
            'observed': observed if observed is not None else self.pending,
            'at_end': self.color,
            'hold_frames': self.hold,
            'streak': self.streak,
            'since_seconds': self.since if display == self.color else self.last_stable_since,
        }
