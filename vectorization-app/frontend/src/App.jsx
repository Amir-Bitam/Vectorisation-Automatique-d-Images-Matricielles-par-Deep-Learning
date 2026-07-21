import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

import ComparisonViewer from "./components/ComparisonViewer";
import UploadPage from "./components/UploadPage";
import {
  DEFAULT_NUM_REGIONS,
  MAX_NUM_REGIONS,
  MIN_NUM_REGIONS,
  getAbsoluteUrl,
  getSvgContent,
  isBackendUnavailableError,
  parseNumRegions,
  vectorizeImage,
} from "./services/vectorizationApi";
import { FIT_PADDING, MAX_ZOOM, MIN_ZOOM, ZOOM_STEP, clamp, getZoomedPan } from "./utils/viewerMath";

function App() {
  const inputRef = useRef(null);
  const dragRef = useRef(null);
  const comparisonViewportRef = useRef(null);

  // Core image and result state
  const [selectedFile, setSelectedFile] = useState(null);
  const [numRegions, setNumRegions] = useState(String(DEFAULT_NUM_REGIONS));
  const [originalPreviewUrl, setOriginalPreviewUrl] = useState("");
  const [imageNaturalSize, setImageNaturalSize] = useState({ width: 0, height: 0 });
  const [svgResultUrl, setSvgResultUrl] = useState("");
  const [svgResultContent, setSvgResultContent] = useState("");
  const [svgContentObjectUrl, setSvgContentObjectUrl] = useState("");
  const [resultMeta, setResultMeta] = useState(null);

  // UI state
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isRevectorizeDialogOpen, setIsRevectorizeDialogOpen] = useState(false);
  const [draftNumRegions, setDraftNumRegions] = useState(String(DEFAULT_NUM_REGIONS));
  const [svgLoadFailed, setSvgLoadFailed] = useState(false);

  // Shared comparison viewer state. Both the raster and SVG panels use these values.
  const [viewerZoom, setViewerZoom] = useState(1);
  const [viewerPan, setViewerPan] = useState({ x: 0, y: 0 });

  // Image upload handling: create a browser preview URL for the selected file.
  useEffect(() => {
    if (!selectedFile) {
      setOriginalPreviewUrl("");
      setImageNaturalSize({ width: 0, height: 0 });
      return undefined;
    }

    const nextPreviewUrl = URL.createObjectURL(selectedFile);
    setOriginalPreviewUrl(nextPreviewUrl);

    return () => URL.revokeObjectURL(nextPreviewUrl);
  }, [selectedFile]);

  // Store the real image dimensions so both comparison panels share one canvas size.
  useEffect(() => {
    if (!originalPreviewUrl) {
      setImageNaturalSize({ width: 0, height: 0 });
      return undefined;
    }

    let isCurrent = true;
    const image = new Image();
    image.onload = () => {
      if (!isCurrent) {
        return;
      }

      setImageNaturalSize({
        width: image.naturalWidth || image.width,
        height: image.naturalHeight || image.height,
      });
    };
    image.onerror = () => {
      if (isCurrent) {
        setImageNaturalSize({ width: 0, height: 0 });
      }
    };
    image.src = originalPreviewUrl;

    return () => {
      isCurrent = false;
    };
  }, [originalPreviewUrl]);

  // SVG result handling: turn raw SVG text into an object URL when the backend returns content.
  useEffect(() => {
    if (!svgResultContent) {
      setSvgContentObjectUrl("");
      return undefined;
    }

    const nextSvgUrl = URL.createObjectURL(new Blob([svgResultContent], { type: "image/svg+xml" }));
    setSvgContentObjectUrl(nextSvgUrl);

    return () => URL.revokeObjectURL(nextSvgUrl);
  }, [svgResultContent]);

  const svgDisplayUrl = useMemo(() => {
    if (svgContentObjectUrl) {
      return svgContentObjectUrl;
    }

    if (!svgResultUrl) {
      return "";
    }

    const separator = svgResultUrl.includes("?") ? "&" : "?";
    return `${svgResultUrl}${separator}preview=${resultMeta?.job_id || "svg"}`;
  }, [resultMeta?.job_id, svgContentObjectUrl, svgResultUrl]);

  // SVG download logic: use the generated object URL or backend download URL.
  const downloadHref = svgContentObjectUrl || svgResultUrl;
  const filename = resultMeta?.svg_filename || "vectorized.svg";

  function resetResult() {
    setResultMeta(null);
    setSvgResultUrl("");
    setSvgResultContent("");
    setSvgLoadFailed(false);
    setIsRevectorizeDialogOpen(false);
    setViewerZoom(1);
    setViewerPan({ x: 0, y: 0 });
  }

  function handleSelectedFile(file) {
    if (!file) {
      return;
    }

    setSelectedFile(file);
    setErrorMessage("");
    resetResult();
  }

  function handleFileChange(event) {
    handleSelectedFile(event.target.files?.[0] || null);
    event.target.value = "";
  }

  // Drag and drop handling: reuse the same file selection path as the file input.
  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    handleSelectedFile(event.dataTransfer.files?.[0] || null);
  }

  async function vectorizeCurrentImage({
    numRegionsValue = numRegions,
    preservePreviousResult = false,
  } = {}) {
    setErrorMessage("");

    if (!selectedFile) {
      setErrorMessage("Choose a PNG or JPG image before vectorizing.");
      return false;
    }

    let validatedNumRegions;
    try {
      validatedNumRegions = parseNumRegions(numRegionsValue);
    } catch (error) {
      setErrorMessage(error.message);
      return false;
    }

    // Initial vectorization clears any old result; re-vectorization keeps the old SVG until success.
    if (!preservePreviousResult) {
      resetResult();
    }

    setIsProcessing(true);
    try {
      // API request to FastAPI. Only the SLIC region count varies per request.
      const data = await vectorizeImage({
        file: selectedFile,
        numRegions: validatedNumRegions,
      });

      const nextSvgContent = getSvgContent(data);
      const nextSvgUrl = getAbsoluteUrl(data?.download_url || data?.svg_url || data?.url);

      if (!nextSvgContent && !nextSvgUrl) {
        throw new Error("Vectorization finished, but no SVG file was returned.");
      }

      // Replace the SVG result only after the backend has returned a valid SVG URL or content.
      setResultMeta(data && typeof data === "object" ? data : { svg_filename: "vectorized.svg" });
      setSvgResultContent(nextSvgContent);
      setSvgResultUrl(nextSvgUrl);
      setSvgLoadFailed(false);
      setNumRegions(String(data?.num_regions ?? validatedNumRegions));
      return true;
    } catch (requestError) {
      setErrorMessage(
        isBackendUnavailableError(requestError)
          ? "Backend is not running. Start FastAPI and try again."
          : requestError?.message || "Vectorization failed.",
      );
      return false;
    } finally {
      // Processing/loading state stays active for the whole inference request.
      setIsProcessing(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    await vectorizeCurrentImage();
  }

  // Comparison viewer controls
  const fitView = useCallback(() => {
    const viewport = comparisonViewportRef.current;
    if (!viewport || !imageNaturalSize.width || !imageNaturalSize.height) {
      return;
    }

    const rect = viewport.getBoundingClientRect();
    if (!rect.width || !rect.height) {
      return;
    }

    const fitZoom = Math.min(rect.width / imageNaturalSize.width, rect.height / imageNaturalSize.height) * FIT_PADDING;
    const nextZoom = clamp(fitZoom, MIN_ZOOM, MAX_ZOOM);

    setViewerZoom(nextZoom);
    setViewerPan({
      x: (rect.width - imageNaturalSize.width * nextZoom) / 2,
      y: (rect.height - imageNaturalSize.height * nextZoom) / 2,
    });
  }, [imageNaturalSize.height, imageNaturalSize.width]);

  useEffect(() => {
    if (!downloadHref || !originalPreviewUrl || !imageNaturalSize.width || !imageNaturalSize.height) {
      return undefined;
    }

    const frameId = window.requestAnimationFrame(fitView);
    return () => window.cancelAnimationFrame(frameId);
  }, [downloadHref, fitView, imageNaturalSize.height, imageNaturalSize.width, originalPreviewUrl]);

  const zoomAtPoint = useCallback(
    (anchorX, anchorY, nextZoomValue) => {
      const oldZoom = viewerZoom;
      const newZoom = clamp(nextZoomValue, MIN_ZOOM, MAX_ZOOM);
      if (!Number.isFinite(newZoom) || newZoom === oldZoom) {
        return;
      }

      setViewerPan(getZoomedPan(anchorX, anchorY, oldZoom, newZoom, viewerPan));
      setViewerZoom(newZoom);
    },
    [viewerPan, viewerZoom],
  );

  const zoomAtViewportCenter = useCallback(
    (direction) => {
      const viewport = comparisonViewportRef.current;
      if (!viewport) {
        return;
      }

      const rect = viewport.getBoundingClientRect();
      zoomAtPoint(rect.width / 2, rect.height / 2, viewerZoom + direction * ZOOM_STEP);
    },
    [viewerZoom, zoomAtPoint],
  );

  const zoomIn = useCallback(() => zoomAtViewportCenter(1), [zoomAtViewportCenter]);
  const zoomOut = useCallback(() => zoomAtViewportCenter(-1), [zoomAtViewportCenter]);

  const resetView = useCallback(() => {
    fitView();
  }, [fitView]);

  // Wheel event used for zooming: the point under the mouse stays fixed while zoom changes.
  const handleWheel = useCallback(
    (event) => {
      event.preventDefault();
      const rect = event.currentTarget.getBoundingClientRect();
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;
      const direction = event.deltaY > 0 ? -1 : 1;

      zoomAtPoint(mouseX, mouseY, viewerZoom + direction * ZOOM_STEP);
    },
    [viewerZoom, zoomAtPoint],
  );

  // Pointer events used for panning. Both comparison panels read the same pan state.
  const pointerHandlers = useMemo(
    () => ({
      onPointerDown(event) {
        event.currentTarget.setPointerCapture(event.pointerId);
        dragRef.current = {
          pointerId: event.pointerId,
          startX: event.clientX,
          startY: event.clientY,
          originX: viewerPan.x,
          originY: viewerPan.y,
        };
      },
      onPointerMove(event) {
        if (!dragRef.current || dragRef.current.pointerId !== event.pointerId) {
          return;
        }

        const nextX = dragRef.current.originX + event.clientX - dragRef.current.startX;
        const nextY = dragRef.current.originY + event.clientY - dragRef.current.startY;
        setViewerPan({ x: nextX, y: nextY });
      },
      onPointerUp(event) {
        if (dragRef.current?.pointerId === event.pointerId) {
          dragRef.current = null;
        }
      },
      onPointerCancel(event) {
        if (dragRef.current?.pointerId === event.pointerId) {
          dragRef.current = null;
        }
      },
    }),
    [viewerPan.x, viewerPan.y],
  );

  // New image / reset logic: clear the result and return to the upload page.
  function handleNewImage() {
    setSelectedFile(null);
    setErrorMessage("");
    resetResult();
  }

  function handleOpenRevectorize() {
    setDraftNumRegions(numRegions);
    setErrorMessage("");
    setIsRevectorizeDialogOpen(true);
  }

  function handleCancelRevectorize() {
    if (!isProcessing) {
      setIsRevectorizeDialogOpen(false);
    }
  }

  // Confirm the new SLIC region count before reusing the uploaded image.
  async function handleRevectorize(event) {
    event.preventDefault();
    const succeeded = await vectorizeCurrentImage({
      numRegionsValue: draftNumRegions,
      preservePreviousResult: true,
    });

    if (succeeded) {
      setIsRevectorizeDialogOpen(false);
    }
  }

  if (downloadHref && originalPreviewUrl) {
    return (
      <ComparisonViewer
        contentSize={imageNaturalSize}
        draftNumRegions={draftNumRegions}
        downloadHref={downloadHref}
        errorMessage={errorMessage}
        filename={filename}
        isProcessing={isProcessing}
        isRevectorizeDialogOpen={isRevectorizeDialogOpen}
        maxNumRegions={MAX_NUM_REGIONS}
        minNumRegions={MIN_NUM_REGIONS}
        onCancelRevectorize={handleCancelRevectorize}
        onFit={resetView}
        onNewImage={handleNewImage}
        onOpenRevectorize={handleOpenRevectorize}
        onRevectorize={handleRevectorize}
        onWheel={handleWheel}
        originalPreviewUrl={originalPreviewUrl}
        pan={viewerPan}
        pointerHandlers={pointerHandlers}
        setDraftNumRegions={setDraftNumRegions}
        setSvgLoadFailed={setSvgLoadFailed}
        svgDisplayUrl={svgDisplayUrl}
        svgLoadFailed={svgLoadFailed}
        viewportRef={comparisonViewportRef}
        zoom={viewerZoom}
        zoomIn={zoomIn}
        zoomOut={zoomOut}
      />
    );
  }

  return (
    <UploadPage
      errorMessage={errorMessage}
      handleDrop={handleDrop}
      handleFileChange={handleFileChange}
      handleSubmit={handleSubmit}
      inputRef={inputRef}
      isDragging={isDragging}
      isProcessing={isProcessing}
      maxNumRegions={MAX_NUM_REGIONS}
      minNumRegions={MIN_NUM_REGIONS}
      numRegions={numRegions}
      onDragStateChange={setIsDragging}
      onNumRegionsChange={setNumRegions}
      onPickFile={() => inputRef.current?.click()}
      originalPreviewUrl={originalPreviewUrl}
      selectedFile={selectedFile}
    />
  );
}

export default App;
