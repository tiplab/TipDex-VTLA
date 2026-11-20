"""Lossless RGB transport for the HTTP inference interface."""
import base64
from io import BytesIO
import numpy as np
from PIL import Image


def validate_rgb(value):
    array=np.asarray(value)
    if (array.ndim!=3 or array.shape[-1]!=3 or min(array.shape[:2])<1 or max(array.shape[:2])>4096
            or not np.issubdtype(array.dtype,np.number) or not np.isfinite(array).all()
            or np.any((array<0)|(array>255)) or np.any(array!=np.floor(array))):
        raise ValueError('Images must contain HWC uint8 RGB values, at most 4096x4096')
    return array.astype(np.uint8)


def encode_rgb(rgb):
    """Encode one HWC RGB frame without changing pixel values."""
    stream=BytesIO()
    Image.fromarray(validate_rgb(rgb)).save(stream,format='PNG',compress_level=1)
    return dict(encoding='png',data=base64.b64encode(stream.getvalue()).decode('ascii'))


def decode_rgb(value):
    if not isinstance(value,dict):
        return validate_rgb(value)
    if set(value)!= {'encoding','data'} or value['encoding']!='png' or not isinstance(value['data'],str):
        raise ValueError('Expected a base64 PNG image object')
    if len(value['data'])>24*1024*1024:
        raise ValueError('Encoded image is too large')
    try:
        raw=base64.b64decode(value['data'],validate=True)
        with Image.open(BytesIO(raw)) as image:
            if image.format!='PNG' or image.mode!='RGB' or max(image.size)>4096:
                raise ValueError('Expected an RGB PNG image no larger than 4096x4096')
            return np.array(image,dtype=np.uint8,copy=True)
    except (OSError,TypeError) as error:
        raise ValueError('Invalid PNG image') from error
