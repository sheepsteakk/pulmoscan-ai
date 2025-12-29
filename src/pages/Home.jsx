import React, { useState, useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { ArrowRight, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";

import XRayUploader from "@/components/scanner/XRayUploader";
import AnalysisResult from "@/components/scanner/AnalysisResult";
import InfoSections from "@/components/home/InfoSections";

export default function Home() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const targetRef = useRef(null);

  const { scrollYProgress } = useScroll({
    target: targetRef,
    offset: ["start start", "end start"],
  });

  const y1 = useTransform(scrollYProgress, [0, 1], ["0%", "50%"]);
  const y2 = useTransform(scrollYProgress, [0, 1], ["0%", "-50%"]);
  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);

  const scrollToScanner = () => {
    document.getElementById("scanner")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* HERO */}
      <section
        ref={targetRef}
        className="relative pt-20 pb-32 overflow-hidden bg-white"
      >
        {/* Background gradients */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <motion.div
            style={{ y: y1, opacity }}
            className="absolute top-0 right-0 w-[500px] h-[500px] bg-indigo-50 rounded-full blur-3xl opacity-50 translate-x-1/3 -translate-y-1/3"
          />
          <motion.div
            style={{ y: y2, opacity }}
            className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-blue-50 rounded-full blur-3xl opacity-50 -translate-x-1/3 translate-y-1/3"
          />
        </div>

        <div className="container relative mx-auto px-4 text-center max-w-4xl">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.6 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            {/* Badge */}
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              whileInView={{ scale: 1, opacity: 1 }}
              viewport={{ once: true, amount: 0.6 }}
              transition={{ duration: 0.5 }}
              className="inline-flex items-center rounded-full border border-indigo-100 bg-indigo-50 px-3 py-1 text-sm font-medium text-indigo-600 mb-8"
            >
              <Zap className="mr-2 h-4 w-4 fill-indigo-600 text-indigo-600" />
              Powered by Advanced Deep Learning
            </motion.div>

            {/* Title */}
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-slate-900 mb-6">
              AI Analysis for
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-blue-600">
                Chest X-Rays
              </span>
            </h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true, amount: 0.6 }}
              transition={{ delay: 0.15, duration: 0.8 }}
              className="mt-6 text-xl text-slate-600 leading-relaxed max-w-2xl mx-auto"
            >
              Instant classification and sensitivity mapping for lung X-ray images.
            </motion.p>

            {/* Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.6 }}
              transition={{ delay: 0.25, duration: 0.5 }}
              className="mt-10 flex justify-center gap-4"
            >
              <Button onClick={scrollToScanner} className="h-12 px-8 text-base">
                Start Analysis <ArrowRight className="ml-2 h-4 w-4" />
              </Button>

              <Button
                variant="outline"
                className="h-12 px-8 text-base"
                onClick={() =>
                  document
                    .getElementById("how-it-works")
                    ?.scrollIntoView({ behavior: "smooth" })
                }
              >
                Learn More
              </Button>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* SCANNER */}
      <section
        id="scanner"
        className="relative py-24 bg-slate-50 border-t border-slate-200"
      >
        <div className="container mx-auto px-4">
          {!analysisResult ? (
            <motion.div
              initial={{ opacity: 0, y: 60 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className="max-w-4xl mx-auto"
            >
              <div className="text-center mb-12">
                <h2 className="text-3xl font-bold text-slate-900">Upload Scan</h2>
                <p className="text-slate-500 mt-2">
                  Supported formats: JPEG, PNG
                </p>
              </div>

              <XRayUploader onAnalysisComplete={setAnalysisResult} />
            </motion.div>
          ) : (
            <AnalysisResult
              result={analysisResult}
              onReset={() => setAnalysisResult(null)}
            />
          )}
        </div>
      </section>

      <InfoSections />
    </div>
  );
}
