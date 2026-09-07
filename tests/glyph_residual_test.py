import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")

from vsr_pipeline import Pipeline


def outlined_text():
    image = np.full((110, 330, 3), 160, dtype=np.uint8)
    cv2.putText(image, "SUBTITLE", (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (20, 20, 20), 6, cv2.LINE_8)
    cv2.putText(image, "SUBTITLE", (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 2, cv2.LINE_8)
    return image, [(32, 74, 12, 210)]


def test_mask_includes_dark_letter_outline_without_filling_background():
    original, boxes = outlined_text()
    pipe = Pipeline.__new__(Pipeline)
    mask = pipe.propainter_boxes_to_mask(boxes, original, (0, 110, 0, 330))
    dark = original[:, :, 0] == 20
    assert np.count_nonzero(mask[dark]) / np.count_nonzero(dark) > 0.95
    assert mask[35, 200] == 0 and mask[72, 100] == 0


def test_residual_check_catches_dark_outline_after_white_fill_is_removed():
    original, boxes = outlined_text()
    fixed = original.copy()
    fixed[np.all(fixed == 255, axis=2)] = 160
    residual = Pipeline.__new__(Pipeline)._residual_mask(fixed, original, boxes)
    assert np.count_nonzero(residual) >= 50


def test_residual_check_rejects_flat_replacement_and_new_unrelated_edges():
    original, boxes = outlined_text()
    fixed = np.full_like(original, 160)
    cv2.line(fixed, (220, 40), (290, 70), (20, 20, 20), 3)
    residual = Pipeline.__new__(Pipeline)._residual_mask(fixed, original, boxes)
    assert not residual.any()


def test_residual_check_rejects_new_white_background_in_original_text_area():
    original, boxes = outlined_text()
    fixed = np.full_like(original, 255)
    residual = Pipeline.__new__(Pipeline)._residual_mask(fixed, original, boxes)
    assert not residual.any()


@pytest.mark.parametrize("bright", [False, True])
def test_residual_check_rejects_unrelated_line_crossing_original_letters(bright):
    original, boxes = outlined_text()
    fixed = np.full_like(original, 160)
    if bright:
        fixed[50:60, 15:200] = 255
    else:
        cv2.line(fixed, (15, 55), (200, 55), (20, 20, 20), 3)
    residual = Pipeline.__new__(Pipeline)._residual_mask(fixed, original, boxes)
    assert not residual.any()


@pytest.mark.parametrize("internal_blend", [False, True])
def test_compositing_does_not_turn_a_new_background_line_into_a_repair(internal_blend):
    original, boxes = outlined_text()
    fixed = np.full_like(original, 160)
    cv2.line(fixed, (15, 55), (200, 55), (20, 20, 20), 3)
    pipe = Pipeline.__new__(Pipeline)
    mask = pipe.propainter_boxes_to_mask(boxes, original, (0, 110, 0, 330))
    calls = []

    def inpaint(frames, masks):
        calls.append(len(frames))
        if internal_blend:
            # The real engine composites against a mask dilated by four pixels.
            cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            return [np.where(cv2.dilate(m, cross, iterations=4)[:, :, None] > 0,
                             fixed, f) for f, m in zip(frames, masks)]
        return [fixed.copy() for _ in frames]

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    output, repairs = pipe._repair_propainter_segment([original] * 2, [mask] * 2, [boxes] * 2)
    assert repairs == 0 and len(calls) == 1
    assert all(np.array_equal(frame[mask == 0], original[mask == 0]) for frame in output)


@pytest.mark.parametrize("frame,roi", [(60, (455, 545, 165, 585)), (686, (830, 935, 180, 575))])
def test_real_visible_subtitle_is_reported(frame, roi):
    directory = os.environ.get("VSR_DIAGNOSTIC_DIR")
    if not directory:
        pytest.skip("set VSR_DIAGNOSTIC_DIR to the original/result frame audit")
    folder = Path(directory)
    with (folder / "probe.json").open() as source:
        record = next(item for item in json.load(source) if item["frame"] == frame)
    original = cv2.imread(str(folder / f"source-{frame:04d}.png"))
    result = cv2.imread(str(folder / f"result-{frame:04d}.png"))
    residual = Pipeline.__new__(Pipeline)._residual_mask(result, original, record["source_boxes"])
    y1, y2, x1, x2 = roi
    assert np.count_nonzero(residual[y1:y2, x1:x2]) >= 50
