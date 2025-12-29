import React from "react";

export function Badge({ className = "", ...props }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium ${className}`}
      {...props}
    />
  );
}
