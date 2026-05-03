export const MIN_ZOOM = 0.05;
export const MAX_ZOOM = 8;
export const ZOOM_STEP = 0.2;
export const FIT_PADDING = 0.9;

export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

// Keep the point under the cursor fixed while the zoom value changes.
export function getZoomedPan(anchorX, anchorY, oldZoom, newZoom, oldPan) {
  return {
    x: anchorX - ((anchorX - oldPan.x) / oldZoom) * newZoom,
    y: anchorY - ((anchorY - oldPan.y) / oldZoom) * newZoom,
  };
}
