import densecap.util


def test_IoU_overlap():
    box0 = (0, 0, 100, 100)
    box1 = (50, 50, 100, 100)
    box2 = (0, 50, 100, 100)
    box3 = (50, 0, 100, 100)

    assert densecap.util.iou(box0, box1) == 50 ** 2 / (2 * 100 ** 2 - 50 ** 2)
    assert densecap.util.iou(box0, box2) == (50 * 100) / (2 * 100 ** 2 - 50 * 100)
    assert densecap.util.iou(box0, box3) == (50 * 100) / (2 * 100 ** 2 - 50 * 100)


def test_IoU_no_overlap():
    box1 = (0, 0, 100, 100)
    box2 = (100, 100, 20, 20)
    box3 = (0, 120, 40, 40)
    assert densecap.util.iou(box1, box2) == 0.0
    assert densecap.util.iou(box1, box3) == 0.0


def test_IoU_inside():
    box1 = (0, 0, 100, 100)
    box2 = (10, 10, 20, 20)
    box3 = (0, 30, 40, 40)
    assert densecap.util.iou(box1, box2) == 20 ** 2 / 100 ** 2
    assert densecap.util.iou(box2, box1) == 20 ** 2 / 100 ** 2
    assert densecap.util.iou(box3, box1) == 40 ** 2 / 100 ** 2


def test_IoU_with_self():
    box1 = (0, 0, 10, 10)
    assert densecap.util.iou(box1, box1) == 1.0
