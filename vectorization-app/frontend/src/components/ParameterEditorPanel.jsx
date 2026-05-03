import React from "react";
import { RefreshCcw, SlidersHorizontal } from "lucide-react";

function ParameterEditorPanel({
  draftOptimizeIter,
  draftPathNum,
  isProcessing,
  onCancel,
  onRevectorize,
  setDraftOptimizeIter,
  setDraftPathNum,
}) {
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-slate-950/40 backdrop-blur-sm">
      <button
        className="hidden flex-1 cursor-default lg:block"
        type="button"
        aria-label="Close parameter settings"
        onClick={onCancel}
        disabled={isProcessing}
      />
      <aside className="flex h-full w-full max-w-md flex-col border-l border-slate-200 bg-white shadow-2xl">
        <form className="flex h-full flex-col" onSubmit={onRevectorize}>
          <div className="border-b border-slate-200 px-6 py-5">
            <div className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-lg bg-violet-50 text-violet-700">
                <SlidersHorizontal className="h-5 w-5" aria-hidden="true" />
              </div>
              <div>
                <h2 className="text-lg font-black text-slate-950">Edit parameters</h2>
                <p className="text-sm text-slate-500">Re-run SuperSVG with the same image.</p>
              </div>
            </div>
          </div>

          <div className="grid flex-1 content-start gap-5 overflow-y-auto px-6 py-6">
            <label className="grid gap-2">
              <span className="text-sm font-bold text-slate-800">path_num</span>
              <input
                className="h-11 rounded-lg border border-slate-300 bg-white px-3 text-sm font-semibold text-slate-900 outline-none transition focus:border-blue-500 focus:ring-4 focus:ring-blue-100"
                type="number"
                step="1"
                value={draftPathNum}
                onChange={(event) => setDraftPathNum(event.target.value)}
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
                value={draftOptimizeIter}
                onChange={(event) => setDraftOptimizeIter(event.target.value)}
                disabled={isProcessing}
              />
              <span className="text-xs leading-5 text-slate-500">Number of optimization iterations, 0 means faster</span>
            </label>
          </div>

          <div className="grid gap-3 border-t border-slate-200 px-6 py-5 sm:grid-cols-2">
            <button
              className="inline-flex h-11 items-center justify-center rounded-lg border border-slate-200 bg-white px-4 text-sm font-black text-slate-700 transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
              type="button"
              onClick={onCancel}
              disabled={isProcessing}
            >
              Cancel
            </button>
            <button
              className="inline-flex h-11 items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-blue-600 to-violet-600 px-4 text-sm font-black text-white shadow-lg shadow-blue-700/20 transition hover:from-blue-700 hover:to-violet-700 disabled:cursor-not-allowed disabled:opacity-60"
              type="submit"
              disabled={isProcessing}
            >
              {isProcessing && <RefreshCcw className="h-4 w-4 animate-spin" aria-hidden="true" />}
              Re-vectorize
            </button>
          </div>
        </form>
      </aside>
    </div>
  );
}

export default ParameterEditorPanel;
