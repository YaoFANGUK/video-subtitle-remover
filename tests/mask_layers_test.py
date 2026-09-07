from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")

from vsr_pipeline import Pipeline


def white_text_frame():
    frame = np.zeros((80, 240, 3), dtype=np.uint8)
    frame[30:42, 20:55] = 255
    frame[28:44, 55:58] = 205
    frame[30:42, 90:125] = 255
    return frame


def test_dual_layer_mask_covers_antialias_but_not_gap():
    pipe = Pipeline.__new__(Pipeline)
    frame = white_text_frame()
    mask = pipe.propainter_boxes_to_mask([(24, 48, 15, 135)], frame, (0, 80, 0, 240))
    assert mask[35, 30] == 255
    assert mask[35, 56] == 255
    assert mask[35, 75] == 0


def test_masks_never_escape_manual_roi():
    pipe = Pipeline.__new__(Pipeline)
    frame = white_text_frame()
    mask = pipe.propainter_boxes_to_mask([(0, 80, 0, 240)], frame, (25, 50, 15, 135))
    assert not mask[:25].any() and not mask[50:].any()
    assert not mask[:, :15].any() and not mask[:, 135:].any()


def test_large_white_object_inside_box_does_not_trigger_rectangle_fallback():
    pipe = Pipeline.__new__(Pipeline)
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    frame[0:160, 80:110] = 255
    mask = pipe.propainter_boxes_to_mask([(50, 75, 10, 200)], frame, (0, 160, 0, 240))
    assert not mask.any()


def test_explicit_wide_sticker_keeps_rectangle():
    pipe = Pipeline.__new__(Pipeline)
    frame = white_text_frame()
    mask = pipe.propainter_boxes_to_mask([], frame, (0, 80, 0, 240),
                                        sticker_boxes=[(24, 48, 15, 135)])
    assert mask[35, 75] == 255


def test_short_white_text_preserves_gap():
    pipe = Pipeline.__new__(Pipeline)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[30:50, 20:35] = 255
    frame[30:50, 50:65] = 255
    mask = pipe.propainter_boxes_to_mask([(25, 55, 15, 75)], frame, (0, 80, 0, 100))
    assert mask[40, 42] == 0


def test_padded_short_ocr_line_keeps_glyph_mask():
    pipe = Pipeline.__new__(Pipeline)
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[20:40, 21:28] = 255
    frame[20:40, 42:50] = 255
    pipe.ocr = SimpleNamespace(predict=lambda img: [{"dt_polys": [
        np.array([[20, 20], [56, 20], [56, 40], [20, 40]])]}])
    region = (0, 60, 0, 80)
    boxes = pipe.detect(frame, region)
    assert boxes
    mask = pipe.propainter_boxes_to_mask(boxes, frame, region)
    assert mask[30, 24] == 255 and mask[30, 35] == 0


def repair_fixture(fail=False):
    pipe = Pipeline.__new__(Pipeline)
    original = white_text_frame()
    box = (24, 48, 15, 135)
    mask = pipe.propainter_boxes_to_mask([box], original, (0, 80, 0, 240))
    calls = []
    first = original.copy()
    first[35, 30] = (245, 245, 245)

    def inpaint(frames, masks):
        calls.append(([f.copy() for f in frames], [m.copy() for m in masks]))
        if len(calls) == 1:
            return [first.copy() for _ in frames]
        if fail:
            raise RuntimeError("out of memory")
        return [np.full_like(f, 77) for f in frames]

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    return pipe, original, box, mask, calls, first


def test_local_repair_uses_first_pass_and_preserves_other_pixels():
    pipe, original, box, mask, calls, first = repair_fixture()
    output, repairs = pipe._repair_propainter_segment([original] * 3, [mask] * 3,
                                                      [[box]] * 3)
    assert repairs == 1 and len(calls) == 2
    assert np.array_equal(calls[1][0][0], first)
    assert np.array_equal(output[0][0, 0], first[0, 0])
    assert np.all(output[0][35, 30] == 77)
    assert np.array_equal(output[0][35, 75], first[35, 75])


def test_failed_local_repair_keeps_first_pass():
    pipe, original, box, mask, calls, first = repair_fixture(fail=True)
    output, repairs = pipe._repair_propainter_segment([original] * 3, [mask] * 3,
                                                      [[box]] * 3)
    assert repairs == 0 and len(calls) == 2
    assert all(np.array_equal(f, first) for f in output)


def test_no_residual_or_disabled_check_does_not_call_second_pass():
    pipe, original, box, mask, calls, first = repair_fixture()
    _, repairs = pipe._repair_propainter_segment([original] * 3, [mask] * 3,
                                                 [[box]] * 3, white_glyph_check=False)
    assert repairs == 0 and len(calls) == 1


def test_residual_outside_effective_mask_does_not_run_empty_second_pass():
    pipe, original, box, mask, calls, first = repair_fixture()
    _, repairs = pipe._repair_propainter_segment(
        [original] * 3, [np.zeros_like(mask)] * 3, [[box]] * 3)
    assert repairs == 0 and len(calls) == 1


def test_residual_check_does_not_mask_new_white_background():
    pipe = Pipeline.__new__(Pipeline)
    original = white_text_frame()
    fixed = np.zeros_like(original)
    fixed[30:42, 65:80] = 255
    residual = pipe._residual_mask(fixed, original, [(24, 48, 15, 135)])
    assert not residual.any()


def test_model_internal_dilation_cannot_change_background_outside_requested_mask():
    pipe = Pipeline.__new__(Pipeline)
    original = white_text_frame()
    box = (24, 48, 15, 135)
    mask = pipe.propainter_boxes_to_mask([box], original, (0, 80, 0, 240))
    pipe.inpainter = SimpleNamespace(inpaint=lambda frames, masks: [
        np.where(cv2.dilate(m, np.ones((9, 9), dtype=np.uint8))[:, :, None] > 0,
                 np.full_like(f, 90), f) for f, m in zip(frames, masks)])
    output, _ = pipe._repair_propainter_segment([original] * 2, [mask] * 2,
                                                [[box]] * 2, white_glyph_check=False)
    assert all(np.array_equal(f[mask == 0], original[mask == 0]) for f in output)


def test_single_frame_segment_supplies_temporal_context_and_returns_one_frame():
    pipe = Pipeline.__new__(Pipeline)
    original = white_text_frame()
    mask = np.full(original.shape[:2], 255, dtype=np.uint8)

    def inpaint(frames, masks):
        assert len(frames) >= 2, "RAFT requires a frame pair"
        assert len(frames) == len(masks)
        return [np.zeros_like(f) for f in frames]

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    result, repairs = pipe._repair_propainter_segment([original], [mask], [[]])
    assert len(result) == 1 and repairs == 0
