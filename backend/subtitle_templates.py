"""Bounded, content-verified recovery of subtitle glyph masks."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class _Template:
    frame_number: int
    box: tuple
    glyph: np.ndarray
    edges: np.ndarray
    support: np.ndarray
    contrast: np.ndarray
    core: np.ndarray
    pixels: int


class SubtitleTemplates:
    """Keep local glyph references; the caller resets at scene boundaries.

    Coordinates are half-open ``(y1, y2, x1, x2)``. Only original input
    masks become references, so inferred pixels cannot reinforce themselves.
    """

    _CACHE_LIMIT = 64

    def __init__(self, region, max_age=60):
        self.region = tuple(int(value) for value in region)
        self.max_age = max(0, int(max_age))
        self._templates = []
        self.stats = {"matched_boxes": 0, "recovered_frames": 0,
                      "added_pixels": 0, "cached_templates": 0, "cached_bytes": 0}

    def reset(self):
        self._templates.clear()
        self._update_cache_stats()

    def _clip_box(self, box, shape):
        y1, y2, x1, x2 = (int(value) for value in box)
        ry1, ry2, rx1, rx2 = self.region
        y1, y2 = max(0, ry1, y1), min(shape[0], ry2, y2)
        x1, x2 = max(0, rx1, x1), min(shape[1], rx2, x2)
        return (y1, y2, x1, x2) if y1 < y2 and x1 < x2 else None

    @staticmethod
    def _edges(frame):
        return cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 60, 140)

    @staticmethod
    def _contrast(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return gray - cv2.GaussianBlur(gray, (0, 0), 1)

    def _remember(self, frame, mask, box, frame_number):
        y1, y2, x1, x2 = box
        glyph = np.where(mask[y1:y2, x1:x2] > 0, 255, 0).astype(np.uint8)
        pixels = int(np.count_nonzero(glyph))
        # Filled fallback rectangles and very small fragments are not references.
        if pixels < 48 or pixels > glyph.size * 0.90:
            return
        patch = frame[y1:y2, x1:x2]
        white = (patch.min(axis=2) > 220) & (np.ptp(patch, axis=2) < 40)
        white_pixels = int(np.count_nonzero(white))
        # A partial mask must not hide the changed suffix during content checks.
        if white_pixels >= 24 and np.count_nonzero(white & (glyph > 0)) < 0.9 * white_pixels:
            return
        support = cv2.dilate(glyph, np.ones((5, 5), dtype=np.uint8))
        edges = cv2.bitwise_and(self._edges(patch), support)
        if np.count_nonzero(edges) < 24:
            return
        contrast = self._contrast(patch) * (glyph > 0)
        core = (white & (glyph > 0)).astype(np.uint8)
        reference = _Template(frame_number, box, glyph, edges, support, contrast, core, pixels)
        for index, previous in enumerate(self._templates):
            if previous.box != box:
                continue
            same_observation = previous.frame_number == frame_number
            same_content = (np.array_equal(previous.glyph, glyph)
                            and np.array_equal(previous.edges, edges))
            if same_observation or same_content:
                if pixels >= previous.pixels and frame_number >= previous.frame_number:
                    self._templates[index] = reference
                return
        self._templates.append(reference)
        if len(self._templates) > self._CACHE_LIMIT:
            self._templates.sort(key=lambda item: (item.pixels, item.frame_number), reverse=True)
            del self._templates[self._CACHE_LIMIT:]

    @staticmethod
    def _nearby(source, target):
        sy1, sy2, sx1, sx2 = source
        ty1, ty2, tx1, tx2 = target
        sh, th = sy2 - sy1, ty2 - ty1
        sw, tw = sx2 - sx1, tx2 - tx1
        return (0.5 <= th / sh <= 2.0 and 0.45 <= tw / sw <= 2.2
                and abs((ty1 + ty2) - (sy1 + sy2)) <= max(sh, th)
                and min(sx2, tx2) - max(sx1, tx1) >= 0.5 * min(sw, tw))

    def _align(self, template, frame, target_box):
        if not self._nearby(template.box, target_box):
            return None
        y1, y2, x1, x2 = template.box
        height, width = template.glyph.shape
        radius = min(16, max(4, height // 3))
        search = self._clip_box((y1 - radius, y2 + radius,
                                 x1 - radius, x2 + radius), frame.shape)
        if search is None:
            return None
        sy1, sy2, sx1, sx2 = search
        if sy2 - sy1 < height or sx2 - sx1 < width:
            return None
        edges = self._edges(frame[sy1:sy2, sx1:sx2])
        contrast = self._contrast(frame[sy1:sy2, sx1:sx2])
        scores = cv2.matchTemplate(contrast, template.contrast, cv2.TM_CCORR_NORMED)
        _, score, _, (dx, dy) = cv2.minMaxLoc(scores)
        if not np.isfinite(score) or score < 0.4:
            return None
        observed = edges[dy:dy + height, dx:dx + width]
        near_observed = cv2.dilate(observed, np.ones((3, 3), dtype=np.uint8))
        expected = template.edges > 0
        supported = expected & (near_observed > 0)
        edge_verified = np.count_nonzero(supported) >= 0.88 * np.count_nonzero(expected)
        # A matching prefix cannot authorize copying a changed word or suffix.
        step = max(8, min(24, height // 2))
        for start in range(0, width, step):
            local = expected[:, start:start + step]
            count = np.count_nonzero(local)
            if count >= 8 and np.count_nonzero(supported[:, start:start + step]) < 0.85 * count:
                edge_verified = False
        near_expected = cv2.dilate(template.edges, np.ones((3, 3), dtype=np.uint8))
        relevant = (observed > 0) & (template.support > 0)
        if np.count_nonzero(relevant & (near_expected > 0)) < 0.75 * np.count_nonzero(relevant):
            edge_verified = False
        # Thin shadows retain signed contrast when white clothing hides light edges.
        target_contrast = contrast[dy:dy + height, dx:dx + width] * (template.glyph > 0)
        contrast_verified = self._correlation(template.contrast, target_contrast) >= 0.55
        contrast_step = max(10, min(12, height // 3))
        for start in range(0, width, contrast_step):
            if np.count_nonzero(expected[:, start:start + contrast_step]) < 8:
                continue
            if self._correlation(template.contrast[:, start:start + contrast_step],
                                 target_contrast[:, start:start + contrast_step]) < 0.50:
                return None
        if not (edge_verified or contrast_verified):
            return None
        target_patch = frame[sy1 + dy:sy1 + dy + height, sx1 + dx:sx1 + dx + width]
        if not self._core_consistent(template, target_patch):
            return None
        return sy1 + dy, sx1 + dx

    @staticmethod
    def _core_consistent(template, target_patch):
        target = ((target_patch.min(axis=2) > 220)
                  & (np.ptp(target_patch, axis=2) < 40)).astype(np.uint8)
        gray = cv2.cvtColor(target_patch, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), dtype=np.uint8)
        height, width = target.shape
        outside = cv2.copyMakeBorder((template.glyph > 0).astype(np.uint8),
                                     1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(outside, None, (0, 0), 2)
        background_holes = outside[1:-1, 1:-1] == 0
        for source, other, reverse in [(template.core, target, False),
                                       (target, template.core, True)]:
            near_other = cv2.dilate(other, kernel) > 0
            _, labels, components, _ = cv2.connectedComponentsWithStats(source)
            for index, (x, y, w, h, pixels) in enumerate(components[1:], start=1):
                if pixels < 8 or x == 0 or y == 0 or x + w == width or y + h == height:
                    continue
                component = labels == index
                if reverse:
                    if not np.any(component & (template.support > 0)):
                        continue
                    # Bright background seen through an existing glyph hole is not new ink.
                    if np.count_nonzero(component & background_holes) > 0.5 * pixels:
                        continue
                    ring = (cv2.dilate(component.astype(np.uint8), kernel) > 0) & ~component
                    # A bright connected background is not a reliable target glyph.
                    if not ring.any() or np.median(gray[ring]) > 180:
                        continue
                if np.count_nonzero(component & near_other) < 0.95 * pixels:
                    return False
        return True

    @staticmethod
    def _correlation(source, target):
        energy = float(np.sum(source * source) * np.sum(target * target))
        return float(np.sum(source * target) / np.sqrt(energy)) if energy > 0 else 0.0

    def _update_cache_stats(self):
        self.stats["cached_templates"] = len(self._templates)
        self.stats["cached_bytes"] = sum(
            item.glyph.nbytes + item.edges.nbytes + item.support.nbytes
            + item.contrast.nbytes + item.core.nbytes
            for item in self._templates)

    def refine(self, frames_bgr, masks, boxes, frame_numbers):
        """Return new ROI-clipped masks and boxes for an overlapping frame window."""
        length = len(frames_bgr)
        if not (len(masks) == len(boxes) == len(frame_numbers) == length):
            raise ValueError("frames, masks, boxes and frame_numbers must have equal lengths")
        if not length:
            return [], []
        numbers = [int(number) for number in frame_numbers]
        oldest = min(numbers) - self.max_age
        self._templates = [item for item in self._templates if item.frame_number >= oldest]
        new_masks, new_boxes = [], []
        for frame, mask, frame_boxes, number in zip(frames_bgr, masks, boxes, numbers):
            if frame.shape[:2] != mask.shape:
                raise ValueError("frame and mask dimensions must match")
            clipped_mask = np.zeros(mask.shape, dtype=np.uint8)
            region = self._clip_box(self.region, mask.shape)
            if region is not None:
                y1, y2, x1, x2 = region
                clipped_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            clipped_boxes = [valid for box in frame_boxes
                             if (valid := self._clip_box(box, mask.shape)) is not None]
            new_masks.append(clipped_mask)
            new_boxes.append(clipped_boxes)
            for box in clipped_boxes:
                self._remember(frame, clipped_mask, box, number)
        # Register the window first: a later clear frame can repair an earlier one.
        for frame, mask, frame_boxes, number in zip(frames_bgr, new_masks, new_boxes, numbers):
            before = int(np.count_nonzero(mask))
            for box_index, box in enumerate(frame_boxes):
                candidates = sorted(
                    (item for item in self._templates
                     if abs(number - item.frame_number) <= self.max_age
                     and item.frame_number != number and self._nearby(item.box, box)),
                    key=lambda item: (-item.pixels, abs(item.frame_number - number)))
                for template in candidates:
                    location = self._align(template, frame, box)
                    if location is None:
                        continue
                    y1, x1 = location
                    height, width = template.glyph.shape
                    current = mask[y1:y1 + height, x1:x1 + width]
                    added = (template.glyph > 0) & (current == 0)
                    if added.any():
                        current[:] = np.maximum(current, template.glyph)
                        ys, xs = np.nonzero(added)
                        frame_boxes[box_index] = (
                            min(box[0], y1 + int(ys.min())),
                            max(box[1], y1 + int(ys.max()) + 1),
                            min(box[2], x1 + int(xs.min())),
                            max(box[3], x1 + int(xs.max()) + 1))
                        self.stats["matched_boxes"] += 1
                    break
            added_pixels = int(np.count_nonzero(mask)) - before
            self.stats["added_pixels"] += added_pixels
            self.stats["recovered_frames"] += int(added_pixels > 0)
        self._update_cache_stats()
        return new_masks, new_boxes
