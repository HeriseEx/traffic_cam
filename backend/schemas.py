from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Point = tuple[Unit, Unit]
Violation = Literal['UNKNOWN', 'NONE', 'SOLID_LINE', 'WRONG_WAY', 'RED_LIGHT', 'RESTRICTED_LANE', 'LATERAL_MOVEMENT']


class MobileIncident(BaseModel):
    model_config = ConfigDict(extra='forbid')
    track_id: int = Field(ge=1)
    kind: Literal['RED_LIGHT', 'LATERAL_MOVEMENT']
    start_ms: int = Field(ge=0, le=89000)
    end_ms: int = Field(ge=0, le=89000)
    plate: str | None = Field(default=None, max_length=16, pattern=r'^[\u4e00-\u9fffA-Z0-9·-]*$')
    plate_confirmed: bool = False

    @model_validator(mode='after')
    def interval(self):
        if self.end_ms < self.start_ms:
            raise ValueError('事件结束早于开始')
        if self.plate_confirmed and not self.plate:
            raise ValueError('确认车牌不能为空')
        return self


class MobileCapture(BaseModel):
    """Client hints kept separately from independent server verdicts; track IDs are clip-local."""
    model_config = ConfigDict(extra='forbid')
    camera_mode: Literal['moving'] = 'moving'
    captured_at: float = Field(ge=0, allow_inf_nan=False)
    duration_ms: int = Field(gt=0, lt=89000)
    recording_gaps_ms: int = Field(ge=0, lt=89000)
    prebuffer_truncated: bool = False
    incidents: list[MobileIncident] = Field(default_factory=list, max_length=16)

    @model_validator(mode='after')
    def bounds(self):
        if self.recording_gaps_ms > self.duration_ms or any(i.end_ms > self.duration_ms for i in self.incidents):
            raise ValueError('事件或缓存间隙超出视频范围')
        return self


class Scene(BaseModel):
    model_config = ConfigDict(extra='forbid')
    fixed_camera: bool = False
    solid_line: tuple[Point, Point] | None = None
    allowed_direction: tuple[Point, Point] | None = None
    road_roi: tuple[Unit, Unit, Unit, Unit] = (0, 0, 1, 1)

    @model_validator(mode='after')
    def geometry(self):
        for segment in (self.solid_line, self.allowed_direction):
            if segment and sum((a-b)**2 for a, b in zip(*segment)) < .0025:
                raise ValueError('标定线段太短')
        x1, y1, x2, y2 = self.road_roi
        if x2 <= x1 or y2 <= y1:
            raise ValueError('道路区域无效')
        return self


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    vehicle_model: Literal['yolox_tiny', 'yolox_s', 'yolox_m', 'yolox_l'] = 'yolox_s'
    plate_model: Literal['hyperlpr3', 'ppocrv5_mobile', 'ppocrv5_server', 'onnxocr_plate'] = 'hyperlpr3'
    vehicle_threshold: float = Field(default=.35, ge=.1, le=.9, allow_inf_nan=False)
    sample_fps: float = Field(default=2, ge=1, le=5, allow_inf_nan=False)
    plate_enabled: bool = True
    plate_threshold: float = Field(default=.85, ge=.5, le=.99, allow_inf_nan=False)
    plate_min_hits: int = Field(default=2, ge=1, le=8)
    threads: int = Field(default=4, ge=1, le=8)
    rules_enabled: bool = True


class Review(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_revision: int = Field(ge=1)
    decision: Literal['VALID', 'INVALID', 'UNCERTAIN', 'RESET']
    reviewer: str = Field(min_length=1, max_length=60)
    note: str = Field(default='', max_length=1000)
    plate: str | None = Field(default=None, max_length=16, pattern=r'^[\u4e00-\u9fffA-Za-z0-9·-]*$')
    violation_type: Violation | None = None

    @model_validator(mode='after')
    def reason(self):
        if self.decision in ('INVALID', 'UNCERTAIN') and not self.note.strip():
            raise ValueError('请填写干预原因')
        if not self.reviewer.strip():
            raise ValueError('请填写复核人')
        if self.plate is not None:
            self.plate = self.plate.upper()
        return self


class Reanalyze(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_revision: int = Field(ge=1)
    config: AnalysisConfig | None = None
    scene: Scene | None = None


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_revision: int = Field(ge=1)
    config: AnalysisConfig
    password: SecretStr = Field(default=SecretStr(''), max_length=128, repr=False)


class Submission(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_revision: int = Field(ge=1)
    receipt: str = Field(min_length=1, max_length=200)
