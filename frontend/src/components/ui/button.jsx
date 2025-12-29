import React from "react";

export function Button({
  className = "",
  variant = "solid",
  ...props
}) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-xl px-6 py-3 text-sm font-semibold transition focus:outline-none";

  const variants = {
    solid:
      "bg-gradient-to-r from-indigo-600 to-indigo-500 text-white shadow-lg shadow-indigo-500/25 hover:-translate-y-0.5 hover:shadow-indigo-500/35",
    outline:
      "border border-slate-200 bg-white text-slate-900 shadow-sm hover:bg-slate-50",
  };

  return (
    <button
      className={`${base} ${variants[variant]} ${className}`}
      {...props}
    />
  );
}
