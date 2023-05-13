# from transformers import LayoutLMv3FeatureExtractor
from PIL import Image

# block 1: shared functions: normalize bbox, normalize segment boxes, load images;

def _get_line_bbox(bboxs):
    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]
    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
    return bbox
def _normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]
def _load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)

# double checked
def _extend_shared_bbox(doc_dict):
    block_ids = doc_dict['block_ids']
    tboxes = doc_dict['tboxes']
    doc_dict['bboxes'] = []
    if not tboxes or len(block_ids)!=len(tboxes):
        return

    block_num = block_ids[0]  # 11
    window_bboxes = [tboxes[0]]
    l = 0
    for r in range(1,len(block_ids)):
        curr_seg = block_ids[r]
        if curr_seg!=block_num:
            new_bboxes = _get_line_bbox(window_bboxes)
            doc_dict['bboxes'] += new_bboxes
            # reset the params
            l = r
            block_num = curr_seg
            window_bboxes = [tboxes[r]]
        else:
            window_bboxes.append(tboxes[r])
    # process the last one
    new_bboxes = _get_line_bbox(window_bboxes)
    doc_dict['bboxes'] += new_bboxes
    return doc_dict

# def _pixel_feature(image_path):
#     feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained('/home/ubuntu/resources/layoutlmv3.base',apply_ocr=False)
#     image, size = _load_image(image_path)
#     encoding = feature_extractor(image)
#     pixel_values = encoding.pixel_values[0]
#     return pixel_values
