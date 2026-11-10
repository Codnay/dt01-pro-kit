/* Map view.
 *
 * One coordinate convention decides everything here. The API doc is explicit:
 * every non-`world` coordinate is a right-handed grid whose origin is the
 * BOTTOM-LEFT of the map image, with angles in radians, counter-clockwise
 * positive. A canvas has its origin top-left and y growing downwards, so the
 * row flip and the heading negation below are not cosmetic — get either wrong
 * and the robot renders mirrored on its own map.
 */

const MapView = (() => {
  const POINT_COLOURS = { init: '#7ab6f5', charge: '#d29922', nav: '#3fb950', inspect: '#b39ce8' };

  let canvas, ctx;
  let image = null;                 // the map PNG, once loaded
  let info = { gridWidth: 0, gridHeight: 0, resolution: 0.05, originX: 0, originY: 0 };
  let view = { scale: 1, x: 0, y: 0, fitted: false };
  let data = { points: [], pose: null, path: [], scan: null };
  let show = { scan: true, path: true };
  let drag = null;

  function init(element, onReadout) {
    canvas = element;
    ctx = canvas.getContext('2d');
    MapView.onReadout = onReadout;

    canvas.addEventListener('mousedown', (event) => {
      drag = { x: event.clientX, y: event.clientY, ox: view.x, oy: view.y };
      canvas.classList.add('dragging');
    });
    window.addEventListener('mousemove', (event) => {
      if (!drag) return;
      view.x = drag.ox + (event.clientX - drag.x);
      view.y = drag.oy + (event.clientY - drag.y);
      draw();
    });
    window.addEventListener('mouseup', () => { drag = null; canvas.classList.remove('dragging'); });

    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      const box = canvas.getBoundingClientRect();
      const px = event.clientX - box.left, py = event.clientY - box.top;
      const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
      const next = Math.max(0.15, Math.min(24, view.scale * factor));
      // Zoom about the cursor rather than the origin.
      view.x = px - (px - view.x) * (next / view.scale);
      view.y = py - (py - view.y) * (next / view.scale);
      view.scale = next;
      draw();
    }, { passive: false });

    canvas.addEventListener('mousemove', (event) => {
      const box = canvas.getBoundingClientRect();
      const grid = toGrid(event.clientX - box.left, event.clientY - box.top);
      if (MapView.onReadout) MapView.onReadout(grid);
    });

    new ResizeObserver(() => resize()).observe(canvas.parentElement);
    resize();
  }

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    const box = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(box.width * ratio));
    canvas.height = Math.max(1, Math.round(box.height * ratio));
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    if (!view.fitted) fit(); else draw();
  }

  /* --- coordinate transforms ------------------------------------------- */

  function toCanvas(gridX, gridY) {
    return { x: view.x + gridX * view.scale,
             y: view.y + (info.gridHeight - gridY) * view.scale };
  }

  function toGrid(canvasX, canvasY) {
    const gx = (canvasX - view.x) / view.scale;
    const gy = info.gridHeight - (canvasY - view.y) / view.scale;
    return {
      gridX: Math.round(gx), gridY: Math.round(gy),
      worldX: info.originX + gx * info.resolution,
      worldY: info.originY + gy * info.resolution,
    };
  }

  function fit() {
    const box = canvas.getBoundingClientRect();
    if (!info.gridWidth || !info.gridHeight || !box.width) return;
    view.scale = Math.min(box.width / info.gridWidth, box.height / info.gridHeight) * 0.92;
    view.x = (box.width - info.gridWidth * view.scale) / 2;
    view.y = (box.height - info.gridHeight * view.scale) / 2;
    view.fitted = true;
    draw();
  }

  /* --- data ------------------------------------------------------------- */

  function setMapInfo(next) {
    if (!next || !next.gridWidth) return;
    const changed = next.gridWidth !== info.gridWidth || next.gridHeight !== info.gridHeight;
    info = Object.assign({}, info, next);
    if (changed) view.fitted = false;
  }

  function loadImage(url) {
    const img = new Image();
    img.onload = () => {
      image = img;
      if (!info.gridWidth) info.gridWidth = img.width;
      if (!info.gridHeight) info.gridHeight = img.height;
      if (!view.fitted) fit(); else draw();
    };
    img.onerror = () => { image = null; draw(); };
    img.src = url;
  }

  function update(next) { Object.assign(data, next); draw(); }
  function toggle(key, value) { show[key] = value; draw(); }

  /* --- drawing ---------------------------------------------------------- */

  function draw() {
    if (!ctx) return;
    const box = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, box.width, box.height);

    if (image) {
      ctx.imageSmoothingEnabled = view.scale < 2;
      const topLeft = toCanvas(0, info.gridHeight);
      ctx.drawImage(image, topLeft.x, topLeft.y,
                    info.gridWidth * view.scale, info.gridHeight * view.scale);
    } else {
      ctx.fillStyle = '#11161e';
      ctx.fillRect(0, 0, box.width, box.height);
      ctx.fillStyle = '#5d6b7c';
      ctx.font = '13px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('no map image', box.width / 2, box.height / 2);
      ctx.textAlign = 'left';
    }

    if (show.path) drawPath();
    if (show.scan) drawScan();
    drawPoints();
    drawRobot();
    drawScaleBar(box);
  }

  function drawPath() {
    const path = data.path || [];
    if (path.length < 2) return;
    ctx.save();
    ctx.strokeStyle = 'rgba(74, 158, 255, .85)';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([7, 5]);
    ctx.beginPath();
    path.forEach((node, index) => {
      const at = toCanvas(node.x, node.y);
      if (index === 0) ctx.moveTo(at.x, at.y); else ctx.lineTo(at.x, at.y);
    });
    ctx.stroke();
    ctx.restore();
  }

  function drawScan() {
    const points = (data.scan && data.scan.gridpoint) || [];
    if (!points.length) return;
    ctx.fillStyle = 'rgba(248, 81, 73, .75)';
    const size = Math.max(1.5, Math.min(4, view.scale * 0.9));
    for (const point of points) {
      const at = toCanvas(point.x, point.y);
      ctx.fillRect(at.x - size / 2, at.y - size / 2, size, size);
    }
  }

  function drawPoints() {
    ctx.font = '600 11px ui-monospace, monospace';
    for (const point of data.points || []) {
      if (point.gridX == null) continue;
      const at = toCanvas(point.gridX, point.gridY);
      const colour = POINT_COLOURS[point.type_name] || '#8b98a8';
      ctx.beginPath();
      ctx.arc(at.x, at.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
      ctx.strokeStyle = '#0d1117';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // Recorded heading: grid y is flipped on canvas, so the angle is too.
      if (typeof point.angle === 'number') {
        ctx.beginPath();
        ctx.moveTo(at.x, at.y);
        ctx.lineTo(at.x + Math.cos(-point.angle) * 13, at.y + Math.sin(-point.angle) * 13);
        ctx.strokeStyle = colour;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      ctx.fillStyle = '#e6edf3';
      ctx.strokeStyle = 'rgba(13,17,23,.9)';
      ctx.lineWidth = 3;
      ctx.strokeText(point.name, at.x + 9, at.y - 7);
      ctx.fillText(point.name, at.x + 9, at.y - 7);
    }
  }

  function drawRobot() {
    const pose = data.pose;
    if (!pose || !pose.grid || pose.grid.x == null) return;
    const at = toCanvas(pose.grid.x, pose.grid.y);
    const theta = (pose.world && typeof pose.world.theta === 'number')
      ? pose.world.theta : (pose.angle || 0);
    const heading = -theta;                       // canvas y grows downwards

    ctx.save();
    ctx.translate(at.x, at.y);
    ctx.rotate(heading);
    ctx.beginPath();                              // a chevron, so heading is unmistakable
    ctx.moveTo(14, 0); ctx.lineTo(-9, 9); ctx.lineTo(-4, 0); ctx.lineTo(-9, -9);
    ctx.closePath();
    ctx.fillStyle = '#4a9eff';
    ctx.fill();
    ctx.strokeStyle = '#0d1117';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();

    if (data.alarm) {                             // proximity ring while something is close
      ctx.beginPath();
      ctx.arc(at.x, at.y, 20, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(248, 81, 73, .85)';
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }
  }

  function drawScaleBar(box) {
    if (!info.resolution || !view.scale) return;
    const metresPerPixel = info.resolution / view.scale;
    let metres = 1;
    while (metres / metresPerPixel > 130) metres /= 2;
    while (metres / metresPerPixel < 40) metres *= 2;
    const width = metres / metresPerPixel;
    const y = box.height - 16, x = box.width - width - 16;
    ctx.strokeStyle = 'rgba(230,237,243,.75)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y - 4); ctx.lineTo(x, y); ctx.lineTo(x + width, y); ctx.lineTo(x + width, y - 4);
    ctx.stroke();
    ctx.fillStyle = 'rgba(230,237,243,.75)';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(metres >= 1 ? `${metres} m` : `${metres * 100} cm`, x + width / 2, y - 7);
    ctx.textAlign = 'left';
  }

  return { init, fit, setMapInfo, loadImage, update, toggle, resize };
})();
