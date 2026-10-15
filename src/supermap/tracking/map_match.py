"""Map-guided association of leftover D_high detections (Sec. IV-B: unique instance ids via 3D consistency)."""
from __future__ import annotations

from supermap.tracking.state import box_to_state


# REQ-022, REQ-026: leftover D_high detections take the id of the O_on object whose footprint IoU >= thresh.
# Duplicate boxes of an object already matched this frame are dropped; returns the detections left for new tracks.
def match_leftovers(tr, high, rem_high, boxes, scores, labels, just_removed, out) -> list[int]:
    from supermap.tracking.bytetrack import Track, TrackState, iou_matrix  # deferred: avoids import cycle
    ids = sorted(tr._map_boxes)
    if not ids or not rem_high:
        return list(rem_high)
    order = sorted(rem_high, key=lambda k: -scores[high[k]])
    iou = iou_matrix(boxes[[high[k] for k in order]], [tr._map_boxes[i] for i in ids])
    rest: list[int] = []
    for row, k in enumerate(order):
        d = high[k]
        j = int(iou[row].argmax())
        if iou[row, j] < tr.map_match_iou:
            rest.append(k)
            continue
        oid = ids[j]
        trk = tr.tracks.get(oid)
        if trk is not None and trk.last_matched_frame == tr.frame:
            tr.map_dropped += 1   # duplicate box of an object already matched in this frame
            continue
        if trk is None and oid in just_removed:
            trk = tr.removed.pop(oid)
            tr.tracks[oid] = trk
        if trk is None:
            old = tr.removed.get(oid)
            trk = Track(oid, box_to_state(boxes[d], tr.dim), tr.Sigma0.copy(),
                        old.label if old is not None else int(labels[d]), float(scores[d]),
                        old.start_frame if old is not None else tr.frame, tr.frame,
                        TrackState.TRACKED, reactivated=True)
        out.append(tr._apply_match(trk, d, boxes, scores, labels))
        tr.map_matched += 1
    return rest
