def find_limits(lbl, step, margin):
    has_label = lambda bins: len(bins) > 1 or bins[0].item() != 0
    h, w = lbl.shape[1:]
    up, down, left, right = -1, -1, -1, -1
    for x1 in range(step, w, step):
        if has_label(lbl[:, :, x1 - step:x1].unique()):
            left = max(0, x1 - (step + margin))
            break
    for x2 in range(step, w, step):
        if has_label(lbl[:, :, w - x2:w - x2 + step].unique()):
            right = min(w, w - x2 + (step + margin))
            break
    for y1 in range(step, h, step):
        if has_label(lbl[:, y1 - step:y1, :].unique()):
            up = max(0, y1 - (step + margin))
            break
    for y2 in range(step, h, step):
        if has_label(lbl[:, h - y2:h - y2 + step, :].unique()):
            down = min(h, h - y2 + (step + margin))
            break
    assert up != -1 and down != -1 and left != -1 and right != -1
    return up, down, left, right
