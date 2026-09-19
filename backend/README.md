# 本机车辆筛查工作台 · v2

独立后端服务，配合 `../IllegalCapture` 安卓应用。FastAPI + SQLite WAL + ONNX Runtime CPU + 单个 worker；无需 Redis 或前端构建工具。

**默认自动上传、分析并同步结果，人工干预是可选操作。** 尚未提交的记录可纠正车牌、修改判定、标记 AI 无效或撤销干预；AI 原始结果保留，并记录修改历史。已提交记录通过服务端校验锁定，不能复核或重跑。当前未接入外部举报平台，`submission_allowed=false`。

## 2026-09 行车取证链路更新

新版 App 提供轨迹编号、短时遮挡保留、按发送帧关联 OCR、按事件持续时间动态截取、后台上传重试以及片段预览/筛选/本地删除。具体参数和边界见 [App README](../IllegalCapture/README.md)。下文旧版操作记录中的固定自动片段、按位置投票说明已由该实现替代。

`/health` 新增 `capture_metadata=true`。上传可带 `capture`：移动机位、采集时间、片段时长、分段间隙、缓存不足标志及最多 16 个动作（轨迹编号、类型、片段内起止毫秒、车牌及确认状态）。这些是客户端提示，服务器独立分析，不能直接升级为违法结论。旧服务器仍可接收新版 App 视频。

移动机位明显横移现在返回 `LATERAL_MOVEMENT`，不直接命名为压实线。车牌与动作必须在同一条服务器轨迹上有足够帧数，不再以全片其他车辆的稳定车牌替代。跟踪丢失容忍固定为 3 秒，随配置采样帧率换算。

运行 `python pipeline_smoke.py` 可验证隔离的真机上传链路（61617 端口）；不会更新线上服务器。发布此后端仍使用原有部署流程。

## 启动和使用

```powershell
python download_model.py
docker compose up -d --build --wait
# Ubuntu 一键更新：chmod +x deploy.sh && ./deploy.sh
```

打开工作台 **http://127.0.0.1:61616** 即自动握手登录。Web 采集客户端本机 **https://127.0.0.1:61612**（`python webserve.py`，自签证书点继续；Docker/NPM 仍是 HTTP 61612，线上 **https://cam.muqin.ccwu.cc**）。手机 App 默认 API **https://traffic.muqin.ccwu.cc**。多台可同时在线。客户端用 SHA-256 握手码换会话（`POST /v1/hello`），服务端记录型号和 IP。Nginx Proxy Manager：`traffic.muqin.ccwu.cc` → `127.0.0.1:61616`，`cam.muqin.ccwu.cc` → `127.0.0.1:61612`，并转发 `X-Real-IP`。接口文档 `/docs`，健康检查 `/health`。

1. 左侧按状态、提交/复核状态和车牌或事件编号查找记录；选择后预览视频、检测框和车牌证据，点击证据卡定位视频时间。
2. “AI 原始判断”和“当前生效结果”分别显示自动结果和人工覆盖。完成分析后可在“人工干预”填写原因并保存，手机自动接收变化；无需逐条人工确认后才能处理。
3. “模型与参数”可切换 YOLOX-Tiny / S / M / L，调整车辆阈值、采样帧率、车牌阈值、一致帧数、CPU 线程，并开关车牌识别和违法候选规则。
4. 配置对新任务生效；历史任务保留当时的配置。展开“道路标定”，可以对未提交记录按当前配置重新分析，旧结果存入审计历史。
5. 页面每 4 秒刷新；正在编辑的表单不会被新结果覆盖。并发修改会返回冲突并要求加载最新版本。

预览是单独的 H.264 浏览器兼容副本，原始视频及 SHA-256 不变；识别框只在页面叠加，不写进原片。支持视频范围请求和拖动播放。

## 安卓连接

```powershell
adb reverse tcp:61616 tcp:61616
```

手机点画面显示菜单 → “服务器与记录”，默认 `https://traffic.muqin.ccwu.cc`，打开即自动登录。本机联调改为 `http://127.0.0.1:61616`。令牌不必填写。重连 USB 后可能需要重新执行 ADB 转发。

- 按钮“标记重点”，或开启语音后说 **开始标记**，导出触发前约 10–15 秒缓存 + 触发后 10 / 15 / 30 秒无声片段，完成后自动入队上传。稳定红灯且前车继续接近、车牌多帧一致时 App 也会自动截取（`trigger=automatic`）。录制期间继续识别车辆。
- 离线片段保存在手机私有目录，连接恢复后自动重试；服务器确认接收后删除待上传副本。
- App 前台约每 3 秒同步结果和人工修改；离开前台暂停轮询，再打开时按持久化游标补齐。当前没有后台推送服务。
- 连接后约每 2.5 秒发送一张 JPEG（质量 95）到本机后端，不保存该帧；响应含车牌和 `signal_observed`。手机按车牌位置对最近约 4 次结果投票，纠正省份汉字单帧误读。语音完全离线，不发送或保存音频。

## 车牌与违法候选的边界

HyperLPR3 检测车牌、透视矫正、CTC 识别大陆车牌，按置信度和格式过滤。视频默认 2 fps，至少 2 个采样帧的文字一致才成为主车牌；单帧候选仍可在证据卡查看。原始分辨率用于 OCR 裁剪，不能先缩到 720p：实测缩小后可能漏字符且置信度仍很高。

1080p 整帧缩到 640 检测会直接漏掉远处小车牌（行车记录仪画面前车车牌约 70px，缩后仅 23px）。识别现在是整帧一遍 + 大图 2×2 重叠分块，检出的小车牌再按原图邻域放大重读；同一块车牌的多个读数按“合法字符数多者优先、再比置信度”合并——掉字是已确认的失败模式（苏ED51712 曾被读成苏ED5112），多字未见。单帧省份汉字在小车牌上仍可能读错（实测 川 误读为 冀），多帧一致性才是可信度依据：夜间行车记录仪两段 20 秒 1080p 片段实测（`validation/dashcam_clips_check.py`），主车牌分别为 绿牌 川AA91307（20/40 帧，conf 0.9984）和 蓝牌 川A8BX43（25/40 帧，conf 0.9998），省份单帧误读均为少数票；注意同一物理车牌的误读文字命中 ≥2 帧时也会标记 stable，主车牌以命中数排序为准。大角度侧向车牌（左前车 B39131）仍会全程漏检。

多帧一致只是可信度提示，不保证正确。遮挡、远距离、小车牌、夜间、运动模糊仍需真实道路样本评估；同一静态图片重复多帧不代表运动场景验证。一个视频可能包含多个车牌，页面保留全部候选。

当前优先完善车牌和灯色。服务器对每段视频输出 `signal_state`（红/黄/绿/灭/无法确认），连续 3 个采样帧同色才切换，单帧闪光不得翻转。夜间 1080p 实测 YOLOX 交通灯类分数低于 0.1，因此用画面上部饱和色块找灯芯（`signals.py`，`signal_model=HSV_GLOW`）；路灯大光斑会被面积过滤掉，白天过亮的画面不走该启发式。`test.mp4` 00:00 路口段 1.5 秒进入稳定红灯。另输出 `signal_approach`：红灯期间前车底边是否下移，**移动机位不能据此判定越线**。

三条件同时成立时写入 `violations` 一条 `RED_LIGHT` 候选（`decision=CANDIDATE`，`reason=RED_LIGHT_CANDIDATE`）：稳定红灯、红灯期间继续接近、车牌多帧一致。这不是法律证明，`submission_allowed` 仍为 false。实时接口 `POST /v1/recognize-frame` 同样返回 `lights` 与 `signal_observed`，供手机 HUD 和自动截取；关闭车牌识别时仍检测灯色。

违法规则另外提供 **固定机位 + 手工标定实线/行驶方向 + 背景稳定** 时的疑似压线、逆行候选；移动镜头、背景不足或缺少标定返回无法可靠判定。不实现越线或专用车道自动判断。规则不具备可直接举报的证据资格。

手机在相机开启时滚动缓存 5 秒短片段；标记或自动触发后拼接触发前约 10–15 秒。离开画面会停缓存。自动化已覆盖片段入队后的上传、分析、回传，人工不是必经审核环节。`validation/red_light_loop_check.py` 用 `test.mp4` 00:00 段跑本地分析并走 Docker 上传/复核。

## API

业务接口要求 Bearer token；网页登录换取 8 小时 HttpOnly、SameSite=Strict 会话。Cookie 写操作另校验 `X-Requested-With: traffic-console`，不开放跨域访问。时间为 Unix 秒，`revision` 用于防止覆盖并发修改。

| 接口 | 用途 |
| --- | --- |
| `POST /v1/tasks` | 上传原始视频，首次 202，幂等重传 200 |
| `GET /v1/tasks` | 分页、状态、复核/提交状态、关键词过滤 |
| `GET /v1/tasks/{id}` | 完整结果和采样帧证据 |
| `GET /v1/tasks/{id}/video` | 浏览器预览；`?original=true` 获取原片 |
| `GET /v1/changes?after=游标` | 增量结果；不含逐帧框，支持断线补齐 |
| `POST /v1/tasks/{id}/review` | `expected_revision`、decision、reviewer、note、可选 plate/violation_type |
| `POST /v1/tasks/{id}/reanalyze` | 使用配置/道路标定重跑；保留旧结果审计 |
| `GET /v1/tasks/{id}/audit` | 修改历史 |
| `GET /v1/overview` | 状态统计、worker 心跳 |
| `GET/PUT /v1/settings` | 模型目录、运行参数；PUT 需要 expected_revision |
| `POST /v1/recognize-frame` | 不落盘的 JPEG；返回车牌（若开启）、`lights`、`signal_observed`；最大 2 MiB / 4K |
| `POST /v1/tasks/{id}/retry` | ERROR 任务重试，最多 3 次 |
| `DELETE /v1/tasks/{id}` | 清理视频，记录保留，处理中返回 409 |
| `POST /v1/tasks/{id}/submission-receipt` | 外部提交成功后记录凭证并锁定；此接口本身不执行提交 |

上传头：`Content-Type: video/mp4`、`X-Video-SHA256: 原片哈希`、`X-Event-Metadata: JSON`。元数据必填 UUID `event_id`，可选 `trigger=manual/voice/automatic/import`、`trigger_text`、`candidate_type`、`app_version`、`model_version`、`scene`。含中文的 HTTP 头 JSON 使用 ASCII `\u` 转义。

同一事件 ID 的视频哈希和规范化元数据必须相同，否则 409。视频最大 50 MiB、90 秒、4K；非法输入 422、超限 413、鉴权失败 401。状态为 `QUEUED → PROCESSING → ANALYZED / REJECTED / ERROR`，过期为 `EXPIRED`。分析完成不等于违法成立。

## 数据、部署和恢复

`data/tasks.sqlite3` 保存队列、结果、配置、修订号和审计；`data/videos/` 保存原片，`data/previews/` 保存预览。默认从上传起保留视频 72 小时；worker 清理视频后仍保留结果。原片落盘后才入队，每帧续租，进程中断后 300 秒租约到期可回收。预留磁盘 256 MiB。

宿主机 API 监听 `61616`（容器内 8000），Web 客户端监听 `61612`。局域网手机开相机请用 **https://电脑IP:61612**（自签证书点继续；页面把 `/v1` 转到 API，避免 HTTPS 页请求 HTTP）。远程用 Nginx Proxy Manager：`https://traffic.muqin.ccwu.cc` → `http://127.0.0.1:61616`，`https://cam.muqin.ccwu.cc` → `http://127.0.0.1:61612`，并转发真实 IP。客户端打开后自动 `POST /v1/hello`（Android / iOS / web 同一套 SHA-256 握手，`code = sha256(device_id + 换行 + platform + 换行 + ts + 换行 + nonce + 换行 + traffic-hello-v1)`）。线上默认连 `https://traffic.muqin.ccwu.cc`。Ubuntu 上在本目录执行 `chmod +x deploy.sh && ./deploy.sh`。迁移时一起保留 `data/`、`models/`、`.env`。`.env` 不应提交、嵌入 APK 或写进日志。

`docker compose stop` 停止，`docker compose up -d --wait` 恢复。保留时间和磁盘限制见 `config.py` / `compose.yaml`；模型参数以 UI 保存的配置为准。

## 可重复验证

本地 Python 3.13、FFmpeg/ffprobe；依赖见固定版本 `requirements.txt`。

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_backend.py
.\.venv\Scripts\python.exe plate_smoke.py
# 可选：重建容器后的排队恢复验证
.\.venv\Scripts\python.exe smoke.py --restart
```

安卓构建后运行：

```powershell
Set-Location ..\IllegalCapture
.\gradlew.bat :app:assembleDebug :app:assembleDebugAndroidTest :app:lintDebug
Set-Location ..\backend
.\.venv\Scripts\python.exe android_smoke.py --adb "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
```

2026-09-17 本版验证：后端 4 个集成测试通过，覆盖实际 S/Tiny 推理、中文 OCR、规则门控、登录和写入保护、乐观锁、人工修改/撤销与重跑审计、提交锁定、增量同步、视频范围请求和旧版任务恢复。实际 Docker 图片及视频 OCR 样例识别为 `苏ED51712`，视频为 6 个采样帧。

小米 M2007J3SC / Android 12 八个真机测试全部通过：车辆模型、车牌按中心点关联车辆框、车牌位置投票纠错、灯色保持与前车接近门控、无声片段拼接、横竖屏及环形缓存真实录制上传（预热 12 秒后片段 ≥14 秒）、后台纠正自动回传、麦克风开启及真实离线中文 ASR。语音测试使用本机合成的“开始标记”PCM，不代表车内噪声或远场唤醒效果。UI 已验证预览、证据定位、人工纠正、原始结果保留和模型切换。

报告在 `validation/android-tests.txt`、`android-result.json`、`plate-result.json`，界面截图同目录。脚本仅通过标准输入临时提供测试令牌，结束后删除。测试会创建本机样例业务记录。

## 模型与许可

- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)，v0.1.1rc0 Tiny / S / M / L ONNX，Apache-2.0；许可 `models/LICENSE-YOLOX.txt`。官方没有更新的预训练权重；M/L 是同一发布里更大的 COCO 模型。
- [HyperLPR3](https://github.com/szad670401/HyperLPR)，基于 0.1.3 预处理和 CTC 解码适配，20230229 detector + recognizer ONNX，Apache-2.0；许可 `models/plate/LICENSE.txt`。上游 20230229 仍是最新公开 ONNX，没有可替换的新权重。
- 固定 SHA-256 在 `model_catalog.py`；加载和下载均校验。官方 HyperLPR 包以 HTTP 发布，下载脚本对压缩包和每个权重分别校验固定哈希。
- [Vosk 中文小模型](https://alphacephei.com/vosk/models)，small-cn-0.22，Apache-2.0，打包在安卓 assets；`python download_model.py --android-voice` 可恢复。
- 测试图片/视频来源见 [tests/README.md](tests/README.md)。
