(function () {
  const host = window.parent || window;
  const root = host.document;
  if (!root || !root.body) return;

  const NS = "__kbDockManagerStableV3";
  if (host[NS] && typeof host[NS].destroy === "function") {
    try { host[NS].destroy(); } catch (e) {}
  }

  const RESIZE_CLASS = "kb-resizing";
  const DOCK_SIDE_GAP = 35;
  const DOCK_RIGHT_GAP = 35;
  const MIN_WIDTH = 320;
  const state = {
    raf: 0,
    timer: 0,
    ro: null,
    mo: null,
    dragging: false,
    form: null,
    ta: null,
    onMouseDown: null,
    onPointerDown: null,
    onTouchStart: null,
    onMouseMove: null,
    onPointerMove: null,
    onMouseUp: null,
    onPointerUp: null,
    onPointerCancel: null,
    onTouchEnd: null,
    onKeyDown: null,
    onBlur: null,
    onResize: null,
  };

  function isInsideStaleNode(el) {
    try { return !!(el && el.closest && el.closest('[data-stale="true"]')); } catch (e) { return false; }
  }
  function pickFresh(nodes) {
    for (const n of (nodes || [])) {
      if (!n) continue;
      if (!isInsideStaleNode(n)) return n;
    }
    return (nodes && nodes.length) ? nodes[0] : null;
  }
  function findMainRegion() {
    return pickFresh(root.querySelectorAll('section.main'));
  }
  function findMainContainer() {
    return pickFresh([
      ...root.querySelectorAll('section.main .block-container'),
      ...root.querySelectorAll('[data-testid="stMainBlockContainer"]'),
      ...root.querySelectorAll('.block-container')
    ]);
  }
  function findSidebar() {
    return pickFresh(root.querySelectorAll('section[data-testid="stSidebar"]'));
  }
  function _btnText(btn) {
    return String((btn && (btn.innerText || btn.textContent)) || "").trim().toLowerCase();
  }
  function isStopBtnText(t) {
    const s = String(t || "").trim().toLowerCase();
    return s === "■" || s === "停止" || s === "stop";
  }
  function isSendBtnText(t) {
    const s = String(t || "").trim().toLowerCase();
    return s === "发送" || s === "↑" || s === "send" || s === "submit";
  }
  function hasSendButton(form) {
    if (!form) return false;
    const btns = form.querySelectorAll('button');
    for (const b of btns) {
      if (isInsideStaleNode(b)) continue;
      const txt = _btnText(b);
      if (isStopBtnText(txt)) continue;
      if (isSendBtnText(txt)) return true;
    }
    return false;
  }
  function clickSendButton(form) {
    if (!form) return false;
    const btns = Array.from(form.querySelectorAll("button"));
    let fallback = null;
    for (const b of btns) {
      if (!b || isInsideStaleNode(b) || b.disabled) continue;
      const txt = _btnText(b);
      if (isStopBtnText(txt)) continue;
      if (isSendBtnText(txt)) {
        b.click();
        return true;
      }
      if (!fallback) {
        const typ = String(b.getAttribute("type") || "").toLowerCase();
        if (typ === "submit" || b.closest('div[data-testid="stFormSubmitButton"]')) {
          fallback = b;
        }
      }
    }
    if (fallback) {
      fallback.click();
      return true;
    }
    return false;
  }
  function findPromptFormAndTextarea() {
    const forms = root.querySelectorAll('div[data-testid="stForm"], form');
    for (const form of forms) {
      if (isInsideStaleNode(form)) continue;
      const ta =
        form.querySelector('div[data-testid="stTextArea"] textarea') ||
        form.querySelector('.stTextArea textarea') ||
        form.querySelector('textarea');
      if (!ta || isInsideStaleNode(ta)) continue;
      if (hasSendButton(form)) return { form, ta };
    }
    return { form: null, ta: null };
  }
  function setResizing(on) {
    if (on) root.body.classList.add(RESIZE_CLASS);
    else root.body.classList.remove(RESIZE_CLASS);
  }
  function placeDock(form) {
    if (!form) return;
    const mainContainer = findMainContainer();
    const mainRegion = findMainRegion();
    const sidebar = findSidebar();
    const anchor = mainContainer || mainRegion;
    if (!anchor && !sidebar) return;

    const viewportW = Math.max(0, (host.innerWidth || root.documentElement.clientWidth || 0));
    const anchorRect = (anchor && anchor.getBoundingClientRect) ? anchor.getBoundingClientRect() : null;
    const sidebarRect = (sidebar && sidebar.getBoundingClientRect) ? sidebar.getBoundingClientRect() : null;

    let leftBound = DOCK_SIDE_GAP;
    let rightBound = Math.max(leftBound + MIN_WIDTH, viewportW - DOCK_RIGHT_GAP);

    if (anchorRect && isFinite(anchorRect.left) && isFinite(anchorRect.right) && anchorRect.width > 10) {
      leftBound = Math.max(leftBound, Math.floor(anchorRect.left) + DOCK_SIDE_GAP);
      rightBound = Math.min(rightBound, Math.floor(anchorRect.right) - DOCK_SIDE_GAP);
    }
    if (sidebarRect && isFinite(sidebarRect.right) && sidebarRect.width > 10) {
      leftBound = Math.max(leftBound, Math.floor(sidebarRect.right) + DOCK_SIDE_GAP);
    }
    if (!isFinite(leftBound) || !isFinite(rightBound)) return;

    rightBound = Math.min(rightBound, viewportW - DOCK_RIGHT_GAP);
    if (rightBound - leftBound < MIN_WIDTH) rightBound = leftBound + MIN_WIDTH;
    if (rightBound > viewportW - DOCK_RIGHT_GAP) {
      rightBound = viewportW - DOCK_RIGHT_GAP;
      leftBound = Math.max(DOCK_SIDE_GAP, rightBound - MIN_WIDTH);
    }

    const dockLeft = Math.max(DOCK_SIDE_GAP, Math.floor(leftBound));
    const dockWidth = Math.max(MIN_WIDTH, Math.floor(rightBound - dockLeft));

    form.classList.add('kb-input-dock', 'kb-dock-positioned');
    form.style.left = dockLeft + 'px';
    form.style.right = 'auto';
    form.style.width = dockWidth + 'px';
    form.style.transform = 'none';
  }
  function bindCtrlEnter(ta, form) {
    if (!ta || ta.dataset.kbCtrlEnterHooked === "1") return;
    ta.dataset.kbCtrlEnterHooked = "1";
    ta.addEventListener("keydown", function (e) {
      const isCtrlEnter = (e.ctrlKey || e.metaKey) && e.key === "Enter";
      if (!isCtrlEnter) return;
      if (e.isComposing) return;
      const ok = clickSendButton(form || root);
      if (!ok) return;
      e.preventDefault();
      e.stopPropagation();
    }, { capture: true });
  }
  function hook() {
    const hit = findPromptFormAndTextarea();
    if (!hit.form || !hit.ta) return;
    state.form = hit.form;
    state.ta = hit.ta;
    state.form.classList.add("kb-input-dock");
    placeDock(state.form);
    bindCtrlEnter(state.ta, state.form);
  }
  function scheduleHook() {
    if (state.raf) return;
    state.raf = host.requestAnimationFrame(function () {
      state.raf = 0;
      hook();
    });
  }
  function startDragIfNearSidebarEdge(e) {
    const sidebar = findSidebar();
    if (!sidebar || !e) return;
    const clientX = Number(e.clientX);
    if (!isFinite(clientX)) return;
    const rect = sidebar.getBoundingClientRect();
    if (!rect || !isFinite(rect.right)) return;
    const nearEdge = Math.abs(rect.right - clientX) <= 24;
    if (!nearEdge) return;
    state.dragging = true;
    setResizing(true);
    scheduleHook();
  }
  function onDragMove() {
    if (!state.dragging) return;
    scheduleHook();
  }
  function stopDrag() {
    if (!state.dragging) return;
    state.dragging = false;
    setResizing(false);
    scheduleHook();
  }
  function installListeners() {
    state.onMouseDown = startDragIfNearSidebarEdge;
    state.onPointerDown = startDragIfNearSidebarEdge;
    state.onTouchStart = function (e) {
      const t = (e.touches && e.touches[0]) ? e.touches[0] : null;
      if (t) startDragIfNearSidebarEdge(t);
    };
    state.onMouseMove = onDragMove;
    state.onPointerMove = onDragMove;
    state.onMouseUp = stopDrag;
    state.onPointerUp = stopDrag;
    state.onPointerCancel = stopDrag;
    state.onTouchEnd = stopDrag;
    state.onKeyDown = function (e) {
      try {
        const isCtrlEnter = (e.ctrlKey || e.metaKey) && e.key === "Enter";
        if (!isCtrlEnter || e.isComposing) return;
        const target = e.target;
        if (!target || isInsideStaleNode(target)) return;
        const ta = (target.tagName === "TEXTAREA") ? target : (target.closest ? target.closest("textarea") : null);
        if (!ta) return;
        const hit = findPromptFormAndTextarea();
        if (!hit.form || !hit.ta) return;
        if (ta !== hit.ta && !hit.form.contains(ta)) return;
        const ok = clickSendButton(hit.form);
        if (!ok) return;
        e.preventDefault();
        e.stopPropagation();
      } catch (err) {}
    };
    state.onBlur = stopDrag;
    state.onResize = scheduleHook;

    root.addEventListener("mousedown", state.onMouseDown, true);
    root.addEventListener("pointerdown", state.onPointerDown, true);
    root.addEventListener("touchstart", state.onTouchStart, true);
    root.addEventListener("mousemove", state.onMouseMove, true);
    root.addEventListener("pointermove", state.onPointerMove, true);
    root.addEventListener("mouseup", state.onMouseUp, true);
    root.addEventListener("pointerup", state.onPointerUp, true);
    root.addEventListener("pointercancel", state.onPointerCancel, true);
    root.addEventListener("touchend", state.onTouchEnd, true);
    root.addEventListener("keydown", state.onKeyDown, true);
    host.addEventListener("blur", state.onBlur, true);
    host.addEventListener("resize", state.onResize, { passive: true });
  }
  function installObservers() {
    if (typeof ResizeObserver !== "undefined") {
      try {
        state.ro = new ResizeObserver(function () { scheduleHook(); });
        const candidates = [root.documentElement, root.body, findSidebar(), findMainContainer(), findMainRegion()];
        for (const c of candidates) {
          if (c) state.ro.observe(c);
        }
      } catch (e) {}
    }
    if (typeof MutationObserver !== "undefined") {
      try {
        state.mo = new MutationObserver(function () { scheduleHook(); });
        state.mo.observe(root.body, { childList: true, subtree: true, attributes: true });
      } catch (e) {}
    }
  }
  function destroy() {
    try { if (state.timer) host.clearInterval(state.timer); } catch (e) {}
    try { if (state.raf) host.cancelAnimationFrame(state.raf); } catch (e) {}
    try { if (state.ro) state.ro.disconnect(); } catch (e) {}
    try { if (state.mo) state.mo.disconnect(); } catch (e) {}
    try { root.removeEventListener("mousedown", state.onMouseDown, true); } catch (e) {}
    try { root.removeEventListener("pointerdown", state.onPointerDown, true); } catch (e) {}
    try { root.removeEventListener("touchstart", state.onTouchStart, true); } catch (e) {}
    try { root.removeEventListener("mousemove", state.onMouseMove, true); } catch (e) {}
    try { root.removeEventListener("pointermove", state.onPointerMove, true); } catch (e) {}
    try { root.removeEventListener("mouseup", state.onMouseUp, true); } catch (e) {}
    try { root.removeEventListener("pointerup", state.onPointerUp, true); } catch (e) {}
    try { root.removeEventListener("pointercancel", state.onPointerCancel, true); } catch (e) {}
    try { root.removeEventListener("touchend", state.onTouchEnd, true); } catch (e) {}
    try { root.removeEventListener("keydown", state.onKeyDown, true); } catch (e) {}
    try { host.removeEventListener("blur", state.onBlur, true); } catch (e) {}
    try { host.removeEventListener("resize", state.onResize, false); } catch (e) {}
    setResizing(false);
  }

  host[NS] = { destroy, schedule: scheduleHook };
  installListeners();
  installObservers();
  state.timer = host.setInterval(scheduleHook, 120);
  scheduleHook();
})();
