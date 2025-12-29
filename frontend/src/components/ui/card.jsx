import React from "react";

export function Card({ className = "", children, ...props }) {
  return (
    <div
      className={`
        rounded-2xl
        border border-slate-200
        bg-white
        shadow-[0_10px_25px_rgba(15,23,42,0.08)]
        transition-shadow
        ${className}
      `}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({ className = "", children, ...props }) {
  return (
    <div className={`p-8 pb-0 ${className}`} {...props}>
      {children}
    </div>
  );
}

export function CardTitle({ className = "", children, ...props }) {
  return (
    <h3
      className={`text-xl font-semibold text-slate-900 ${className}`}
      {...props}
    >
      {children}
    </h3>
  );
}

export function CardContent({ className = "", children, ...props }) {
  return (
    <div className={`p-8 pt-0 text-slate-600 leading-relaxed ${className}`} {...props}>
      {children}
    </div>
  );
}
