# 字幕字形补全与残留复核实施计划

> **For agentic workers:** Use superpowers:subagent-driven-development for bounded module work and review. Do not commit or push without the user's request.

**Goal:** 修复 test_pro_16 中的暗描边、白衣服上的半句残留和 OCR 局部漏框，保留人物及背景。

**Architecture:** 保持反馈式 OCR 和 ProPainter 引擎不变。字形层扩展有效擦除边界；短期模板只在文字结构匹配时补全局部遮罩和框；复核比较原文字结构与修复结果，局部二轮后显式报告未解决帧。

**Tech Stack:** Python 3.12、NumPy、OpenCV、PyAV、pytest；GPU 完整画质在服务器验收。

## 约束与批准范围

用户已批准先试用上述方案。隔离工作树基于 06b2ede；不改默认检测范围、不换模型、不增加云端调用、不使用整行矩形作为白字兜底。保留 ROI、切景、60 帧窗口和二轮失败降级。

## Task 1：字形边界和残留结构

状态：实现、需求复审和质量复审通过；补了引擎内部合成负例，
确认背景横线不会因最终裁切触发二轮。已纳入全量 120 项回归。

**Files:** `vsr_pipeline.py`、`tests/mask_layers_test.py`、`tests/glyph_residual_test.py`。

- [x] 写失败测试：深色描边必须覆盖，字间背景不变；原字形位置的暗笔画必须触发复核，新出现的背景边缘不能触发。
- [x] 运行 `python -m pytest -q tests/glyph_residual_test.py` 确認旧实现失败。
- [x] 高置信字形和局部亮边取并集后扩展 2 像素，限制在文字框和 ROI；保留全局大白物体排除。
- [x] `_residual_mask` 在允许字形邻域比较原/结果的亮暗局部对比结构，不再只接受亮度大于 180 的像素；保持背景白块负例。
- [x] 重放第 60、686 帧，并跑原有遮罩测试。

```python
mask = pipe.propainter_boxes_to_mask(boxes, original, region)
assert mask[dark_stroke_y, dark_stroke_x] == 255
assert mask[gap_y, gap_x] == 0
residual = pipe._residual_mask(result_bgr, original_bgr, boxes)
assert np.count_nonzero(residual[text_slice]) >= 50
```

## Task 2：有界跨帧字形模板

状态：实现、最终需求与质量复审通过，27 项模板回归通过（包含真实素材）；
近似字母误匹配和重叠窗口提前淘汰参考均已修复。

**Files:** 创建 `backend/subtitle_templates.py`、`tests/subtitle_templates_test.py`。

- [x] 写测试覆盖部分字形缺失、局部 OCR 框缩水、平移、同位置换字、文字消失、reset、时间过期、ROI 和输入不变性。
- [x] 运行测试观察缺少模块/接口失败。
- [x] 实现 `SubtitleTemplates(region, max_age=60)`，提供 `reset()` 和 `refine(frames_bgr, masks, boxes, frame_numbers)`，返回新的 `(masks, boxes)`。
- [x] 只缓存有限的局部图像/字形，不缓存全片；通过 OpenCV 文字边缘及局部对比度核实内容、估计小幅平移；仅几何重叠不允许复用。
- [x] 匹配成功时只传播字形形状，修正局部缩短的框；无当前文字框/结构不匹配时不传播。切景由调用方 reset，时间差由帧号限制。
- [x] 遮罩始终限制 ROI，支持 40+20 窗口重叠的重复帧号。

```python
templates = SubtitleTemplates(region)
refined_masks, refined_boxes = templates.refine(frames, masks, boxes, frame_numbers)
templates.reset()
```

## Task 3：接入、验收与日志

状态：主集成需求与质量复审通过；最终复核异常不再阻断写出，另计
`residual_check_failed`。全量 120 项通过，整方案最终需求与质量审查通过。

**Files:** `vsr_pipeline.py`、`tests/adaptive_detection_test.py`、使用文档。

- [x] 写集成测试固定模板在二轮前生效、切景重置、未解决残留日志以及非遮罩区域不变。
- [x] 每次 process_video 创建独立模板实例；ProPainter 分段送入前补全字形。合成和二轮使用补全后的有效遮罩/文字框。
- [x] 二轮结束后重查残留，打印未解决帧数量和补全统计；不做无上限循环修复。
- [x] 生成真实 2 秒、4 秒、结尾的遮罩覆盖对照；真实素材路径通过环境变量指定，常规测试不依赖本机视频。
- [x] 全量 pytest、py_compile、git diff --check；独立需求审查后做代码质量审查。
- [x] 更新试跑命令及验证限制，保留 GPU 全流程待验收状态，不声称画质已修复。

## 实际验证结果

- 全量：120 passed；设置 `VSR_DIAGNOSTIC_DIR` 与 `VSR_TEMPLATE_REGRESSION_VIDEO`，包含四项真实采样回归。
- 遮罩对照：`/private/tmp/vsr-result-analysis.aSCqKp/glyph-fix-{0060,0120,0686}.png`。
- 第 60 帧：本次扩边后 4118 像素，使用第 30 帧参考补为 8517 像素。
- 第 120 帧：本次扩边后 15962 像素，使用第 180 帧参考补为 16738 像素，恢复圆点和 S。该回放验证可用参考下的能力，不保证默认窗口能访问第 180 帧。
- 第 686 帧：旧遮罩 10060 像素，本次扩边后 16000 像素；新复核在旧成片字幕带检测到 447 个疑似残留像素。
- 未运行 CUDA ProPainter 全片推理，GPU 画质、显存及耗时验收仍待服务器试跑。

## 验证命令

```bash
PYTHONPATH=/private/tmp/vsr-test-deps.Lsr98F /Users/liusili/opt/anaconda3/envs/vsr/bin/python -m pytest -q
/Users/liusili/opt/anaconda3/envs/vsr/bin/python -m py_compile vsr_pipeline.py backend/subtitle_templates.py
git diff --check
```
