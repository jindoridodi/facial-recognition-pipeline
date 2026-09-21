// Expose the tracker in browsers and CommonJS without duplicating its implementation.
(function attachFaceTracker(root, factory) {
  // The tracker is used directly by the browser and imported by Node's unit tests.
  const exports = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = exports;
  }
  if (root) {
    root.FaceTracker = exports.FaceTracker;
  }
}(typeof globalThis !== 'undefined' ? globalThis : this, function createFaceTracker() {
  // Build the module's private helpers before returning its public tracker class.
  function copyBox(box) {
    // Normalize server-provided box fields before using them in geometry calculations.
    return {
      x: Number(box.x),
      y: Number(box.y),
      width: Number(box.width),
      height: Number(box.height)
    };
  }

  function center(box) {
    // Return a box midpoint for motion estimation and deterministic screen ordering.
    return {
      x: box.x + box.width / 2,
      y: box.y + box.height / 2
    };
  }

  function predictedBox(track) {
    // Estimate the next position with the track's most recently observed velocity.
    return {
      ...track.box,
      x: track.box.x + track.velocity.x,
      y: track.box.y + track.velocity.y
    };
  }

  function intersectionOverUnion(first, second) {
    // Measure box overlap on a 0–1 scale for association scoring.
    const left = Math.max(first.x, second.x);
    const top = Math.max(first.y, second.y);
    const right = Math.min(first.x + first.width, second.x + second.width);
    const bottom = Math.min(first.y + first.height, second.y + second.height);
    const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
    const union = first.width * first.height + second.width * second.height - intersection;
    return union > 0 ? intersection / union : 0;
  }

  function normalizedCenterDistance(first, second) {
    // Compare midpoint separation relative to face size so the threshold scales naturally.
    const firstCenter = center(first);
    const secondCenter = center(second);
    const distance = Math.hypot(
      firstCenter.x - secondCenter.x,
      firstCenter.y - secondCenter.y
    );
    const firstDiagonal = Math.hypot(first.width, first.height);
    const secondDiagonal = Math.hypot(second.width, second.height);
    return distance / Math.max(firstDiagonal, secondDiagonal, 1);
  }

  function subjectId(identity) {
    // Support both current identity responses and the legacy name-only representation.
    return identity?.subject_id || identity?.name || null;
  }

  class FaceTracker {
    // Associate per-frame detector output with stable IDs for one camera session.
    constructor({
      maxMissedFrames = 5,
      minIntersectionOverUnion = 0.1,
      maxCenterDistance = 1.5
    } = {}) {
      // Configure how long tracks persist and how permissive geometric matching is.
      this.maxMissedFrames = maxMissedFrames;
      this.minIntersectionOverUnion = minIntersectionOverUnion;
      this.maxCenterDistance = maxCenterDistance;
      this.reset();
    }

    reset() {
      // Discard session state when a camera starts or stops.
      this.tracks = new Map();
      this.nextTrackId = 1;
    }

    clearIdentities() {
      // Forget recognized names after all corresponding server enrollments are deleted.
      this.tracks.forEach((track) => { track.subjectId = null; });
    }

    setIdentity(trackId, identity) {
      // Attach an optimistic enrollment result to its stable in-memory track.
      const track = this.tracks.get(trackId);
      if (track) track.subjectId = subjectId(identity);
    }

    update(faces) {
      // Associate this detector frame, advance unmatched tracks, and allocate new IDs.
      const detections = faces.map((face, sourceIndex) => ({
        face,
        sourceIndex,
        box: copyBox(face.box),
        subjectId: subjectId(face.identity)
      }));
      const assignments = new Map();
      const assignedTracks = new Set();
      const assignedDetections = new Set();
      const candidates = [];

      this.tracks.forEach((track) => {
        const predicted = predictedBox(track);
        detections.forEach((detection, detectionIndex) => {
          if (
            track.subjectId
            && detection.subjectId
            && track.subjectId !== detection.subjectId
          ) {
            return;
          }

          const sameSubject = Boolean(
            track.subjectId && detection.subjectId === track.subjectId
          );
          const overlap = intersectionOverUnion(predicted, detection.box);
          const distance = normalizedCenterDistance(predicted, detection.box);
          if (
            !sameSubject
            && overlap < this.minIntersectionOverUnion
            && distance > this.maxCenterDistance
          ) {
            return;
          }

          // Identity is a strong signal; geometry handles anonymous faces and crossings.
          candidates.push({
            track,
            detectionIndex,
            score: (sameSubject ? 4 : 0) + overlap * 2 - distance * 0.25
          });
        });
      });

      candidates.sort((first, second) => (
        // Stable tie-breaks keep IDs reproducible when candidates score equally.
        second.score - first.score
        || first.track.id - second.track.id
        || first.detectionIndex - second.detectionIndex
      ));

      candidates.forEach(({ track, detectionIndex }) => {
        if (assignedTracks.has(track.id) || assignedDetections.has(detectionIndex)) return;
        assignments.set(detectionIndex, track);
        assignedTracks.add(track.id);
        assignedDetections.add(detectionIndex);
      });

      assignments.forEach((track, detectionIndex) => {
        const detection = detections[detectionIndex];
        const oldCenter = center(track.box);
        const newCenter = center(detection.box);
        track.velocity = {
          x: newCenter.x - oldCenter.x,
          y: newCenter.y - oldCenter.y
        };
        track.box = detection.box;
        track.missedFrames = 0;
        if (detection.subjectId) track.subjectId = detection.subjectId;
      });

      this.tracks.forEach((track) => {
        if (assignedTracks.has(track.id)) return;
        // Extrapolate briefly so a dropped detector frame does not create a new ID.
        track.box = predictedBox(track);
        track.missedFrames += 1;
        if (track.missedFrames > this.maxMissedFrames) {
          this.tracks.delete(track.id);
        }
      });

      detections
        .map((detection, detectionIndex) => ({ detection, detectionIndex }))
        .filter(({ detectionIndex }) => !assignedDetections.has(detectionIndex))
        .sort((first, second) => (
          center(first.detection.box).x - center(second.detection.box).x
          || center(first.detection.box).y - center(second.detection.box).y
          || first.detection.sourceIndex - second.detection.sourceIndex
        ))
        .forEach(({ detection, detectionIndex }) => {
          // Detector order is unstable, so assign first-frame IDs in screen order.
          const track = {
            id: this.nextTrackId,
            box: detection.box,
            velocity: { x: 0, y: 0 },
            missedFrames: 0,
            subjectId: detection.subjectId
          };
          this.nextTrackId += 1;
          this.tracks.set(track.id, track);
          assignments.set(detectionIndex, track);
        });

      return detections
        .map((detection, detectionIndex) => ({
          ...detection.face,
          trackId: assignments.get(detectionIndex).id,
          sourceIndex: detection.sourceIndex
        }))
        .sort((first, second) => first.trackId - second.trackId);
    }
  }

  return { FaceTracker };
}));
