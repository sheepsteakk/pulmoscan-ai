import React, { useState, useRef } from "react";
import { UploadCloud, ScanEye } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// Indeterminate (moving) progress bar using Framer Motion
function ProgressBar() {
  return (
    <div className="w-full h-2 rounded-full bg-slate-200 overflow-hidden">
      <motion.div
        className="h-full w-1/3 rounded-full bg-indigo-600"
        animate={{ x: ["-120%", "220%"] }}
        transition={{ duration: 1.15, repeat: Infinity, ease: "easeInOut" }}
      />
    </div>
  );
}

function normalizeBaseUrl(url) {
  return String(url || "")
    .trim()
    .replace(/\/+$/, ""); // remove trailing slashes
}

// 1) Vite env var (only applied at BUILD TIME)
// 2) fallback to your working Render backend
// 3) fallback to local dev
const API_BASE_URL = normalizeBaseUrl(
  import.meta.env.VITE_API_BASE_URL || "https://pulmoscan-ai-ysey.onrender.com"
) || "http://127.0.0.1:8000";

export default function XRayUploader({ onAnalysisComplete }) {
  const [uploading, setUploading] = useState(false);
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const callBackend = async (file) => {
    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const url = `${API_BASE_URL}/predict`;

      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text().catch(() => "");
        throw new Error(
          `Backend error: ${response.status} ${response.statusText}${text ? ` - ${text}` : ""}`
        );
      }

      const data = await response.json();

      const result = {
        prediction: data.prediction,
        confidence: data.confidence,
        imageUrl: data.original_image,
        heatmap: data.heatmap,
      };

      onAnalysisComplete?.(result);
    } catch (err) {
      console.error("Upload/predict failed:", err);
      alert(
        `Backend request failed.\n\nAPI: ${API_BASE_URL}\n\n${err?.message || "Unknown error"}`
      );
    } finally {
      setTimeout(() => setUploading(false), 400);
    }
  };

  const handleFileSelection = async (selectedFile) => {
    if (!selectedFile || uploading) return;
    await callBackend(selectedFile);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    if (!uploading) setIsDragActive(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    setIsDragActive(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragActive(false);
    if (uploading) return;
    const files = e.dataTransfer.files;
    if (files?.length > 0) handleFileSelection(files[0]);
  };

  const onFileInputChange = (e) => {
    const files = e.target.files;
    if (files?.length > 0) handleFileSelection(files[0]);
  };

  const handleDivClick = () => {
    if (!uploading) fileInputRef.current?.click();
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        onClick={handleDivClick}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={`
          relative overflow-hidden rounded-2xl border-2 border-dashed transition-all
          duration-300 ease-in-out cursor-pointer
          ${
            isDragActive
              ? "border-indigo-500 bg-indigo-50"
              : "border-slate-300 hover:border-indigo-400 hover:bg-slate-50"
          }
          ${uploading ? "pointer-events-none opacity-80" : ""}
          h-80 flex flex-col items-center justify-center text-center p-8 bg-white shadow-sm
        `}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={onFileInputChange}
          accept="image/*"
          className="hidden"
        />

        <AnimatePresence mode="wait">
          {uploading ? (
            <motion.div
              key="uploading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="w-full max-w-md space-y-6"
            >
              <div className="relative mx-auto w-20 h-20">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="absolute inset-0 rounded-full border-4 border-indigo-100 border-t-indigo-600"
                />
                <ScanEye className="absolute inset-0 m-auto text-indigo-600 w-8 h-8" />
              </div>

              <div className="space-y-2">
                <h3 className="text-lg font-semibold text-slate-900">
                  Analyzing Radiograph...
                </h3>
                <p className="text-sm text-slate-500">
                  Uploading image and generating occlusion map
                </p>
              </div>

              <ProgressBar />
            </motion.div>
          ) : (
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-4"
            >
              <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-indigo-100 text-indigo-600">
                <UploadCloud className="h-8 w-8" />
              </div>

              <div className="space-y-1">
                <p className="text-lg font-semibold text-slate-900">
                  {isDragActive ? "Drop the X-Ray here" : "Upload Chest X-Ray"}
                </p>
                <p className="text-sm text-slate-500">
                  Drag and drop or click to browse files
                </p>
              </div>

              <div className="flex gap-2 justify-center text-xs text-slate-400 font-mono">
                <span>JPEG</span>
                <span>•</span>
                <span>PNG</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}