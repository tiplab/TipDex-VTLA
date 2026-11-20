import numpy as np
import pytest
from tipdex.wire import encode_rgb,decode_rgb


def test_lossless_transport_preserves_every_pixel():
    rgb=np.random.default_rng(5).integers(0,256,(224,224,3),dtype=np.uint8)
    np.testing.assert_array_equal(decode_rgb(encode_rgb(rgb)),rgb)
    np.testing.assert_array_equal(decode_rgb(rgb.tolist()),rgb)


@pytest.mark.parametrize('value',[{'encoding':'png','data':'not-base64!'},
                                  {'encoding':'jpg','data':''},[[[-1,0,0]]],[[[256,0,0]]]])
def test_invalid_images_are_rejected(value):
    with pytest.raises(ValueError):decode_rgb(value)
