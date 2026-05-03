import React, { useMemo } from "react";
import {
  ArrowLeft,
  Download,
  Maximize2,
  Minus,
  RefreshCcw,
  SlidersHorizontal,
  Wand2,
  ZoomIn,
} from "lucide-react";

import IconButton from "./IconButton";
import ParameterEditorPanel from "./ParameterEditorPanel";
import ProcessingOverlay from "./ProcessingOverlay";

function ComparisonPane({
  alt,
  contentSize = { width: 0, height: 0 },
  label,
  loadFailed,
  onImageError,
  onImageLoad,
  onWheel,
  pointerHandlers,
  src,
  transformStyle,
  viewportRef,
}) {
  const hasContentSize = contentSize.width > 0 && contentSize.height > 0;

  return (
    <section className="flex min-h-[360px] min-w-0 flex-col bg-white md:h-full md:min-h-0">
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-slate-200 px-4">
        <span className="text-sm font-black text-slate-800">{label}</span>
      </div>
      <div
        ref={viewportRef}
        className="relative min-h-[320px] flex-1 overflow-hidden checkerboard md:min-h-0"
        onWheel={onWheel}
        {...pointerHandlers}
      >
        {src && !loadFailed && hasContentSize ? (
          // Both panes render the image inside the same virtual canvas size for accurate comparison.
          <div
            className="absolute left-0 top-0 will-change-transform"
            style={{
              width: `${contentSize.width}px`,
              height: `${contentSize.height}px`,
              ...transformStyle,
            }}
          >
            <img
              className="block h-full w-full select-none object-contain"
              draggable="false"
              src={src}
              alt={alt}
              onError={onImageError}
              onLoad={onImageLoad}
            />
          </div>
        ) : (
          <div className="absolute inset-0 grid place-items-center px-6 text-center">
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-semibold text-rose-800">
              {src && loadFailed
                ? "The generated SVG could not be loaded."
                : "Preparing the comparison view."}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

function ComparisonViewer({
  contentSize,
  draftOptimizeIter,
  draftPathNum,
  downloadHref,
  errorMessage,
  filename,
  isProcessing,
  isSettingsOpen,
  onCancelSettings,
  onEditParameters,
  onFit,
  onNewImage,
  onReset,
  onRevectorize,
  onWheel,
  originalPreviewUrl,
  pan,
  pointerHandlers,
  setDraftOptimizeIter,
  setDraftPathNum,
  setSvgLoadFailed,
  svgDisplayUrl,
  svgLoadFailed,
  viewportRef,
  zoom,
  zoomIn,
  zoomOut,
}) {
  // Comparison viewer rendering: the same transform is applied to both panes.
  const transformStyle = useMemo(
    () => ({
      transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
      transformOrigin: "0 0",
    }),
    [pan.x, pan.y, zoom],
  );

  return (
    <main className="flex min-h-screen flex-col bg-slate-100 text-slate-900">
      <header className="z-20 flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-3 shadow-sm sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-gradient-to-br from-blue-600 to-violet-600 text-white">
            <Wand2 className="h-5 w-5" aria-hidden="true" />
          </div>
          <span className="truncate text-sm font-black text-slate-950 sm:text-base">PFE Vectorization App</span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <IconButton label="Zoom out" onClick={zoomOut}>
            <Minus className="h-4 w-4" aria-hidden="true" />
          </IconButton>
          <div className="flex h-10 min-w-20 items-center justify-center rounded-lg border border-slate-200 bg-slate-50 px-3 text-sm font-black text-slate-700">
            {Math.round(zoom * 100)}%
          </div>
          <IconButton label="Zoom in" onClick={zoomIn}>
            <ZoomIn className="h-4 w-4" aria-hidden="true" />
          </IconButton>
          <IconButton label="Reset view" onClick={onReset}>
            <RefreshCcw className="h-4 w-4" aria-hidden="true" />
          </IconButton>
          <IconButton label="Fit to screen" onClick={onFit}>
            <Maximize2 className="h-4 w-4" aria-hidden="true" />
          </IconButton>
          <button
            className="inline-flex h-10 items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-3 text-sm font-black text-slate-700 shadow-sm transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
            type="button"
            onClick={onEditParameters}
            disabled={isProcessing}
          >
            <SlidersHorizontal className="h-4 w-4" aria-hidden="true" />
            <span className="hidden sm:inline">Edit parameters</span>
          </button>
          {/* SVG download action uses the current generated SVG URL from App.jsx. */}
          <a
            className="inline-flex h-10 items-center justify-center gap-2 rounded-lg bg-emerald-600 px-3 text-sm font-black text-white shadow-sm transition hover:bg-emerald-700"
            href={downloadHref}
            download={filename}
          >
            <Download className="h-4 w-4" aria-hidden="true" />
            <span className="hidden sm:inline">Download SVG</span>
          </a>
          <button
            className="inline-flex h-10 items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-3 text-sm font-black text-slate-700 shadow-sm transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700"
            type="button"
            onClick={onNewImage}
          >
            <ArrowLeft className="h-4 w-4" aria-hidden="true" />
            <span className="hidden sm:inline">New image</span>
          </button>
        </div>
      </header>

      {errorMessage && (
        <div className="border-b border-rose-200 bg-rose-50 px-4 py-3 text-sm font-semibold text-rose-800 sm:px-6 whitespace-pre-wrap">
          {errorMessage}
        </div>
      )}

      <div className="grid flex-1 cursor-grab touch-none select-none overflow-hidden p-3 active:cursor-grabbing md:p-5">
        <div className="grid min-h-[calc(100vh-132px)] overflow-hidden rounded-lg border border-slate-200 bg-white shadow-xl shadow-slate-900/10 md:h-[calc(100vh-132px)] md:min-h-[520px] md:grid-cols-[minmax(0,1fr)_1px_minmax(0,1fr)]">
          <ComparisonPane
            alt="Original raster image"
            contentSize={contentSize}
            label="Original image"
            onWheel={onWheel}
            pointerHandlers={pointerHandlers}
            src={originalPreviewUrl}
            transformStyle={transformStyle}
            viewportRef={viewportRef}
          />
          <div className="hidden bg-slate-300 md:block" aria-hidden="true" />
          <div className="h-px bg-slate-300 md:hidden" aria-hidden="true" />
          <ComparisonPane
            alt="Generated vector SVG"
            contentSize={contentSize}
            label="Vectorized SVG"
            loadFailed={svgLoadFailed}
            onImageError={() => setSvgLoadFailed(true)}
            onImageLoad={() => setSvgLoadFailed(false)}
            onWheel={onWheel}
            pointerHandlers={pointerHandlers}
            src={svgDisplayUrl}
            transformStyle={transformStyle}
          />
        </div>
      </div>

      {/* Settings/Edit parameters modal for re-vectorization without uploading again. */}
      {isSettingsOpen && (
        <ParameterEditorPanel
          draftOptimizeIter={draftOptimizeIter}
          draftPathNum={draftPathNum}
          isProcessing={isProcessing}
          onCancel={onCancelSettings}
          onRevectorize={onRevectorize}
          setDraftOptimizeIter={setDraftOptimizeIter}
          setDraftPathNum={setDraftPathNum}
        />
      )}
      {isProcessing && <ProcessingOverlay />}
    </main>
  );
}

export default ComparisonViewer;
