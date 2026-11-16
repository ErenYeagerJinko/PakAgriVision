"""Tests for the offline LeafSegmenter preprocessing module."""

import numpy as np
from PIL import Image
import pytest

from src.leaf_segmenter import LeafSegmenter, extract_leaf_on_white


def test_leaf_segmenter_initialization():
    segmenter = LeafSegmenter()
    assert segmenter.ready is True
    assert segmenter.session is not None


def test_leaf_segmenter_output_shape_and_color():
    segmenter = LeafSegmenter()
    # Create synthetic test leaf image (green rectangle on brown soil background)
    img = Image.new("RGB", (200, 200), color=(120, 80, 40)) # soil
    for x in range(50, 150):
        for y in range(50, 150):
            img.putpixel((x, y), (30, 160, 40)) # green leaf patch

    composited, mask = segmenter.segment_leaf(img, bg_color=(255, 255, 255))
    assert composited.size == (200, 200)
    assert composited.mode == "RGB"
    assert mask is not None
    assert mask.size == (200, 200)


def test_extract_leaf_on_white_helper():
    img = Image.new("RGB", (100, 100), color=(50, 150, 50))
    result = extract_leaf_on_white(img)
    assert result.size == (100, 100)
    assert result.mode == "RGB"
