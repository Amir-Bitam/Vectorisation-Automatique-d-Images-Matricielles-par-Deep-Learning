import React from "react";
import { RefreshCcw } from "lucide-react";

function ProcessingOverlay() {
  const steps = ["Uploading image", "Running SuperSVG vectorization", "Preparing SVG result"];

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/55 px-4 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-lg border border-white/20 bg-white p-6 shadow-2xl">
        <div className="mx-auto mb-5 grid h-14 w-14 place-items-center rounded-lg bg-blue-50 text-blue-700">
          <RefreshCcw className="h-7 w-7 animate-spin [animation-direction:reverse]" aria-hidden="true" />
        </div>
        <h2 className="text-center text-xl font-bold text-slate-950">Vectorizing your image</h2>
        <p className="mt-2 text-center text-sm leading-6 text-slate-600">
          SuperSVG is tracing paths and optimizing the generated SVG.
        </p>
        <ol className="mt-6 space-y-3">
          {steps.map((step, index) => (
            <li key={step} className="flex items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
              <span className="relative flex h-6 w-6 shrink-0 items-center justify-center">
                <span
                  className="absolute h-6 w-6 animate-ping rounded-full bg-blue-500/20"
                  style={{ animationDelay: `${index * 180}ms` }}
                />
                <span className="grid h-6 w-6 place-items-center rounded-full bg-blue-600 text-xs font-bold text-white">
                  {index + 1}
                </span>
              </span>
              <span className="text-sm font-semibold text-slate-700">{step}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

export default ProcessingOverlay;
