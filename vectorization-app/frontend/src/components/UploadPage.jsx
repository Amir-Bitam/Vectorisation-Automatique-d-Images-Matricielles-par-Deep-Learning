import React from "react";
import { FileImage, ImagePlus, RefreshCcw, UploadCloud, Wand2 } from "lucide-react";

import NavBar from "./NavBar";
import ProcessingOverlay from "./ProcessingOverlay";

function UploadPage({
  errorMessage,
  handleDrop,
  handleFileChange,
  handleSubmit,
  inputRef,
  isDragging,
  isProcessing,
  onDragStateChange,
  onPickFile,
  optimizeIter,
  originalPreviewUrl,
  pathNum,
  selectedFile,
  setOptimizeIter,
  setPathNum,
}) {
  const fileCards = [
    { label: "PNG", tone: "bg-blue-50 text-blue-700 ring-blue-100" },
    { label: "JPG", tone: "bg-violet-50 text-violet-700 ring-violet-100" },
    { label: "JPEG", tone: "bg-emerald-50 text-emerald-700 ring-emerald-100" },
  ];

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <NavBar />

      <form className="mx-auto grid w-full max-w-7xl gap-8 px-5 pb-12 pt-5 sm:px-8 lg:grid-cols-[minmax(0,1.08fr)_380px] lg:items-start lg:pt-10" onSubmit={handleSubmit}>
        <section className="mx-auto w-full max-w-3xl text-center lg:mx-0 lg:max-w-none lg:text-left">
          <div className="mx-auto max-w-3xl lg:mx-0">
            <p className="text-sm font-bold uppercase tracking-[0.18em] text-blue-700">Raster to vector</p>
            <h1 className="mt-4 text-4xl font-black leading-tight text-slate-950 sm:text-5xl lg:text-6xl">
              Convert raster images to vector SVG
            </h1>
            <p className="mx-auto mt-5 max-w-2xl text-base leading-7 text-slate-600 sm:text-lg lg:mx-0">
              Upload a PNG or JPG image and convert it into a clean SVG using SuperSVG.
            </p>
          </div>

          <div
            className={`mt-9 rounded-lg border-2 border-dashed bg-white/90 p-5 shadow-xl shadow-blue-900/10 transition sm:p-7 ${
              isDragging ? "border-blue-500 ring-4 ring-blue-200" : "border-slate-300 hover:border-blue-400"
            }`}
            onDragEnter={(event) => {
              event.preventDefault();
              onDragStateChange(true);
            }}
            onDragOver={(event) => {
              event.preventDefault();
              onDragStateChange(true);
            }}
            onDragLeave={(event) => {
              event.preventDefault();
              if (event.currentTarget === event.target) {
                onDragStateChange(false);
              }
            }}
            onDrop={handleDrop}
          >
            <input ref={inputRef} className="hidden" type="file" accept="image/png,image/jpeg,image/jpg" onChange={handleFileChange} />

            <div className="grid min-h-[360px] place-items-center rounded-lg border border-slate-200 bg-slate-50/80 p-6 checkerboard sm:p-8">
              {originalPreviewUrl ? (
                <div className="grid w-full gap-5">
                  <div className="mx-auto flex h-64 w-full max-w-xl items-center justify-center overflow-hidden rounded-lg border border-slate-200 bg-white shadow-inner">
                    <img className="h-full w-full object-contain" src={originalPreviewUrl} alt="Selected raster preview" />
                  </div>
                  <div className="min-w-0">
                    <p className="truncate text-sm font-bold text-slate-800">{selectedFile?.name}</p>
                    <p className="mt-1 text-sm text-slate-500">Ready to vectorize</p>
                  </div>
                </div>
              ) : (
                <div className="grid justify-items-center gap-5 text-center">
                  <div className="grid h-20 w-20 place-items-center rounded-lg bg-white text-blue-700 shadow-lg shadow-blue-900/10">
                    <UploadCloud className="h-10 w-10" aria-hidden="true" />
                  </div>
                  <div>
                    <p className="text-xl font-bold text-slate-950">Drop your image here</p>
                    <p className="mt-2 text-sm leading-6 text-slate-600">PNG, JPG, and JPEG files are supported.</p>
                  </div>
                </div>
              )}
            </div>

            <div className="mt-6 flex flex-col items-center justify-between gap-4 sm:flex-row">
              <div className="flex flex-wrap justify-center gap-3 sm:justify-start">
                {fileCards.map((card) => (
                  <span key={card.label} className={`inline-flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-black ring-1 ${card.tone}`}>
                    <FileImage className="h-4 w-4" aria-hidden="true" />
                    {card.label}
                  </span>
                ))}
              </div>
              <button
                className="inline-flex h-12 w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-5 text-sm font-bold text-white shadow-lg shadow-blue-700/20 transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto"
                type="button"
                onClick={onPickFile}
                disabled={isProcessing}
              >
                <ImagePlus className="h-5 w-5" aria-hidden="true" />
                Choose image to vectorize
              </button>
            </div>
          </div>
        </section>

        <aside id="settings" className="rounded-lg border border-slate-200 bg-white p-5 shadow-xl shadow-blue-900/10">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-lg bg-violet-50 text-violet-700">
              <Wand2 className="h-5 w-5" aria-hidden="true" />
            </div>
            <div>
              <h2 className="text-lg font-black text-slate-950">Vectorization settings</h2>
              <p className="text-sm text-slate-500">Tune SuperSVG before running.</p>
            </div>
          </div>

          <div className="mt-6 grid gap-5">
            <label className="grid gap-2">
              <span className="text-sm font-bold text-slate-800">path_num</span>
              <input
                className="h-11 rounded-lg border border-slate-300 bg-white px-3 text-sm font-semibold text-slate-900 outline-none transition focus:border-blue-500 focus:ring-4 focus:ring-blue-100"
                type="number"
                step="1"
                value={pathNum}
                onChange={(event) => setPathNum(event.target.value)}
                disabled={isProcessing}
              />
              <span className="text-xs leading-5 text-slate-500">Number of SVG paths to generate</span>
            </label>

            <label className="grid gap-2">
              <span className="text-sm font-bold text-slate-800">optimize_iter</span>
              <input
                className="h-11 rounded-lg border border-slate-300 bg-white px-3 text-sm font-semibold text-slate-900 outline-none transition focus:border-blue-500 focus:ring-4 focus:ring-blue-100"
                type="number"
                step="1"
                value={optimizeIter}
                onChange={(event) => setOptimizeIter(event.target.value)}
                disabled={isProcessing}
              />
              <span className="text-xs leading-5 text-slate-500">Number of optimization iterations, 0 means faster</span>
            </label>
          </div>

          {errorMessage && (
            <div className="mt-5 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-left text-sm font-semibold leading-6 text-rose-800 whitespace-pre-wrap">
              {errorMessage}
            </div>
          )}

          <button
            className="mt-6 inline-flex h-12 w-full items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-blue-600 to-violet-600 px-5 text-sm font-black text-white shadow-lg shadow-blue-700/20 transition hover:from-blue-700 hover:to-violet-700 disabled:cursor-not-allowed disabled:opacity-60"
            type="submit"
            disabled={isProcessing}
          >
            {isProcessing ? <RefreshCcw className="h-5 w-5 animate-spin" aria-hidden="true" /> : <Wand2 className="h-5 w-5" aria-hidden="true" />}
            {isProcessing ? "Processing..." : "Vectorize image"}
          </button>
        </aside>
      </form>

      {isProcessing && <ProcessingOverlay />}
    </main>
  );
}

export default UploadPage;
