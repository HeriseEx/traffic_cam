# 测试媒体

`traffic.jpg`：Richard Croft，2012，*A1 traffic*，[原始来源](https://www.geograph.org.uk/photo/2974654)，[CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/)。未修改图片副本来自 [ageron/data](https://github.com/ageron/data/blob/main/images/traffic.jpg)。

`traffic.mp4`：上述照片生成的 3 秒 H.264 静态视频，10 fps，640×428（底部补一行以满足偶数尺寸），无音频；衍生视频继续按 CC BY-SA 2.0 提供。安卓 assets 中包含同一视频。

在此目录复现：

```powershell
ffmpeg -loop 1 -i traffic.jpg -t 3 -r 10 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -movflags +faststart traffic.mp4
```

它用于真实模型调用和上传链路的可重复验证，不能证明运动车辆跟踪、违法判断或现场识别精度。

`plate.jpg`：HyperLPR 官方仓库的 `resource/images/test_img.jpg`，仓库 Apache-2.0 许可，见 `../models/plate/LICENSE.txt`。来源：https://github.com/szad670401/HyperLPR/blob/master/resource/images/test_img.jpg 。原图未修改，右侧车辆车牌为 `苏ED51712`。

SHA-256：`7253b0e15aff0ffbe19008a58a3d3bcb0322b3a3b03a68423edec728929bc0a6`。

`../plate_smoke.py` 用它生成 `validation/plate-sample.mp4`（3 秒、10 fps、H.264、无音频），验证实时帧和视频多帧 OCR。该衍生测试视频沿用 Apache-2.0，不是实拍违法事件。曾用于定位缩小抽帧导致漏字的问题，当前测试要求保留原始画质后识别完整车牌。
