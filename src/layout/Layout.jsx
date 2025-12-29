import React from "react";
import { Link } from "react-router-dom";
import { Activity, ShieldAlert } from "lucide-react";

export default function Layout({ children }) {
  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 w-full border-b border-slate-200 bg-white/80 backdrop-blur-md">
        <div className="container mx-auto flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">
          <Link
            to="/"
            onClick={(e) => {
              // If you're already on "/", just scroll to top instead of re-navigating
              if (window.location.pathname === "/") {
                e.preventDefault();
              }
              window.scrollTo({ top: 0, left: 0, behavior: "smooth" });
            }}
            className="flex items-center gap-2"
            aria-label="Go to top"
          >
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white">
              <Activity className="h-5 w-5" />
            </div>
            <span className="text-lg font-semibold tracking-tight text-slate-900">
              PulmoScan<span className="text-indigo-600">AI</span>
            </span>
          </Link>
          <div className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600">
            <a href="#scanner" className="hover:text-indigo-600 transition-colors">
              Scanner
            </a>
            <a
              href="#how-it-works"
              className="hover:text-indigo-600 transition-colors"
            >
              How it Works
            </a>
            <a
              href="#disclaimer"
              className="text-indigo-600 hover:text-indigo-700 transition-colors flex items-center gap-1"
            >
              <ShieldAlert className="h-4 w-4" /> Disclaimer
            </a>
          </div>
        </div>
      </nav>

      <main>{children}</main>

      <footer className="border-t border-slate-200 bg-white py-12">
        <div className="container mx-auto px-4 text-center text-slate-500">
          <div className="flex justify-center items-center gap-2 mb-4">
            <Activity className="h-6 w-6 text-indigo-600" />
          </div>
          <p className="text-sm">© 2025 PulmoScan AI. Crafted with passion by Zhen Ying (ɔ◔‿◔)ɔ ♥</p>
        </div>
      </footer>
    </div>
  );
}