import React from "react";

export function Progress({ value = 0, className = "", ...props }) {
  const clamped = Math.max(0, Math.min(100, value));

  return (
    <div
      className={`h-2 w-full rounded-full bg-slate-200 overflow-hidden ${className}`}
      {...props}
    >
      <div
        className="h-full bg-indigo-600 transition-all duration-200"
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
}
