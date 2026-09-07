import json
from types import SimpleNamespace

import numpy as np
import pytest

av = pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")

from vsr_pipeline import Pipeline


def write_frames(path, frames):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264rgb", rate=30)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "rgb24"
        stream.options = {"crf": "0", "bf": "0"}
        for number, image in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = number
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def text_frame():
    frame = np.full((128, 360, 3), 80, dtype=np.uint8)
    cv2.putText(frame, "Sturdy steel", (25, 83), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (15, 15, 15), 6, cv2.LINE_AA)
    cv2.putText(frame, "Sturdy steel", (25, 83), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return frame, (48, 94, 18, 260)


def test_completed_glyphs_reach_model_and_composition_even_from_empty_masks(tmp_path):
    image, box = text_frame()
    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    write_frames(source, [image] * 4)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    pipe.detect = lambda *args: [box]
    complete = pipe.propainter_boxes_to_mask([box], image, (0, 128, 0, 360))
    provided = iter([complete] + [np.zeros_like(complete)] * 3)
    pipe.propainter_boxes_to_mask = lambda *args, **kwargs: next(provided).copy()
    model_masks = []

    def inpaint(frames, masks):
        model_masks.extend(m.copy() for m in masks)
        return [np.full_like(frame, 80) for frame in frames]

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    stats = pipe.process_video(source, output, locate_stickers=False, white_glyph_check=False)
    json.dumps(stats)
    assert len(model_masks) == 4
    assert all(np.array_equal(mask, complete) for mask in model_masks)
    assert stats["template_recovered"] == 3
    with av.open(str(output)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    assert len(frames) == 4
    assert all(np.abs(f[complete > 0].astype(float) - 80).mean() < 3 for f in frames)


def test_unresolved_after_bounded_retry_is_reported(tmp_path, capsys):
    image, box = text_frame()
    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    write_frames(source, [image] * 4)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    pipe.detect = lambda *args: [box]
    calls = []

    def inpaint(frames, masks):
        calls.append(len(frames))
        return [frame.copy() for frame in frames]

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    stats = pipe.process_video(source, output, locate_stickers=False)
    assert len(calls) == 2
    assert stats["unresolved"] == 4
    assert "疑似残留 4" in capsys.readouterr().out


def test_unrecoverable_empty_masks_do_not_load_model(tmp_path):
    image, box = text_frame()
    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    write_frames(source, [image] * 4)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    pipe.detect = lambda *args: [box]
    pipe.propainter_boxes_to_mask = lambda *args, **kwargs: np.zeros(image.shape[:2], np.uint8)
    pipe._ensure_propainter = lambda: pytest.fail("empty masks loaded model")
    stats = pipe.process_video(source, output, locate_stickers=False)
    assert stats["frames"] == 4 and stats["inpainted"] == 0
    assert stats["unresolved"] == 4


def test_final_audit_failure_keeps_first_pass_and_reports_unchecked_frames(tmp_path, capsys):
    image, box = text_frame()
    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    write_frames(source, [image] * 4)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    pipe.detect = lambda *args: [box]
    pipe.inpainter = SimpleNamespace(inpaint=lambda frames, masks: [f.copy() for f in frames])

    def fail_check(*args):
        raise cv2.error("test check failure")

    pipe._residual_mask = fail_check
    stats = pipe.process_video(source, output, locate_stickers=False)
    assert stats["residual_check_failed"] == 4
    assert "复核未完成 4" in capsys.readouterr().out
    with av.open(str(output)) as container:
        assert len(list(container.decode(video=0))) == 4


def test_template_cache_resets_at_scene_boundary(tmp_path, monkeypatch):
    import vsr_pipeline
    from backend.subtitle_templates import SubtitleTemplates

    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    write_frames(source, [np.full((48, 64, 3), value, np.uint8)
                          for value in [0] * 3 + [150] * 3])
    events = []

    class ObservedTemplates(SubtitleTemplates):
        def refine(self, frames, masks, boxes, numbers):
            events.append(list(numbers))
            return super().refine(frames, masks, boxes, numbers)

        def reset(self):
            events.append("reset")
            return super().reset()

    monkeypatch.setattr(vsr_pipeline, "SubtitleTemplates", ObservedTemplates, raising=False)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    pipe.detect = lambda *args: [(20, 30, 10, 50)]
    pipe.inpainter = SimpleNamespace(inpaint=lambda frames, masks: [f.copy() for f in frames])
    pipe.process_video(source, output, locate_stickers=False)
    assert events[-3:] == [[0, 1, 2], "reset", [3, 4, 5]]
