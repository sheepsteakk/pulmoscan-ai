import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { RefreshCw, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

export default function AnalysisResult({ result, onReset }) {
  const [showHeatmap, setShowHeatmap] = useState(true);

  if (!result) return null;

  const isPneumonia = result.prediction === "Pneumonia";
  const confidencePercent = Math.round((result.confidence ?? 0) * 100);

  // Backend returns: imageUrl + heatmap (keep fallbacks)
  const originalSrc =
    result.imageUrl || result.original_image || result.originalImage;
  const heatmapSrc = result.heatmap;

  const imageToShow = useMemo(() => {
    if (showHeatmap && heatmapSrc) return heatmapSrc;
    return originalSrc;
  }, [showHeatmap, heatmapSrc, originalSrc]);

  const downloadCurrentView = () => {
    if (!imageToShow) return;

    const link = document.createElement("a");
    link.href = imageToShow;

    const baseName = "pulmoscan";
    const suffix = showHeatmap ? "heatmap" : "original";
    link.download = `${baseName}-${suffix}.png`;

    link.click();
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-5xl mx-auto mt-12 grid grid-cols-1 lg:grid-cols-2 gap-8"
    >
      {/* Image column */}
      <div className="space-y-4">
        <div className="relative overflow-hidden rounded-2xl border border-slate-200 bg-black shadow-lg aspect-[4/5] sm:aspect-square flex items-center justify-center">
          {imageToShow ? (
            <img
              src={imageToShow}
              alt="X-Ray scan"
              className="h-full w-full object-contain"
            />
          ) : (
            <div className="text-sm text-slate-300">No image available</div>
          )}

          {/* View toggle */}
          <div className="absolute top-3 right-3">
            <div className="inline-flex rounded-xl border border-slate-200 bg-white/90 backdrop-blur px-1 py-1 shadow-sm">
              <button
                type="button"
                onClick={() => setShowHeatmap(false)}
                className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
                  !showHeatmap
                    ? "bg-indigo-600 text-white"
                    : "text-slate-600 hover:text-indigo-600"
                }`}
              >
                Original
              </button>

              <button
                type="button"
                onClick={() => setShowHeatmap(true)}
                disabled={!heatmapSrc}
                className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
                  showHeatmap
                    ? "bg-indigo-600 text-white"
                    : "text-slate-600 hover:text-indigo-600"
                } ${!heatmapSrc ? "opacity-50 cursor-not-allowed" : ""}`}
              >
                Heatmap
              </button>
            </div>
          </div>
        </div>

        <p className="text-xs text-center text-slate-500">
          *Heatmap is a sensitivity overlay for model explainability, not a medical
          diagnosis.
        </p>
      </div>

      {/* Diagnostics column */}
      <div className="flex flex-col justify-center space-y-8">
        <div>
          <Badge
            className={
              isPneumonia
                ? "mb-4 px-4 py-1 text-sm border-red-200 bg-red-50 text-red-700"
                : "mb-4 px-4 py-1 text-sm border-green-200 bg-green-50 text-green-700"
            }
          >
            {isPneumonia ? "Abnormality Detected" : "Normal Reading"}
          </Badge>

          <h2 className="text-4xl font-bold tracking-tight text-slate-900 mb-2">
            {isPneumonia ? "Pneumonia Detected" : "Normal Lung Scan"}
          </h2>
          <p className="text-lg text-slate-600">
            {isPneumonia
              ? "The deep learning model has identified patterns consistent with pneumonia."
              : "No significant abnormalities were detected by the model."}
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Card className="p-5 border-slate-200 bg-slate-50/50">
            <div className="text-sm text-slate-500 mb-1">Confidence Score</div>
            <div
              className={`text-3xl font-bold ${
                isPneumonia ? "text-red-600" : "text-green-600"
              }`}
            >
              {confidencePercent}%
            </div>
            <div className="text-xs text-slate-400 mt-2">Model Certainty</div>
          </Card>

          <Card className="p-5 border-slate-200 bg-slate-50/50">
            <div className="text-sm text-slate-500 mb-1">Scan View</div>
            <div className="text-3xl font-bold text-slate-900">
              Posterior / AP
            </div>
            <div className="text-xs text-slate-400 mt-2">
              Resolution: model-normalized
            </div>
          </Card>
        </div>

        {/* Summary */}
        <div className="space-y-3 pt-4 border-t border-slate-100">
          <h3 className="font-semibold text-slate-900">Analysis Summary</h3>

          <p className="text-sm text-slate-600 leading-relaxed">
            The occlusion sensitivity map shows which lung regions most influenced
            the model’s prediction. Warmer colours indicate higher influence,
            while cooler colours indicate lower influence. This reflects model
            sensitivity, not a medical diagnosis.
          </p>
        </div>

        <div className="flex gap-4 pt-4">
          <Button onClick={onReset} className="flex-1">
            <RefreshCw className="mr-2 h-4 w-4" /> Analyze Another
          </Button>

          <Button
            variant="outline"
            className="flex-1"
            onClick={downloadCurrentView}
            disabled={!imageToShow}
          >
            <Download className="mr-2 h-4 w-4" /> Export{" "}
            {showHeatmap ? "Heatmap" : "Original"}
          </Button>
        </div>
      </div>
    </motion.div>
  );
}