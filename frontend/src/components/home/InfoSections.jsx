import React from "react";
import { motion } from "framer-motion";
import {
  Brain,
  FileSearch,
  Lock,
  ShieldAlert,
  ShieldCheck,
  Stethoscope,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function InfoSections() {
  // Base44-like stagger
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        when: "beforeChildren",
        delayChildren: 0.08,
        staggerChildren: 0.12,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 18 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { type: "spring", stiffness: 260, damping: 22 },
    },
  };

  return (
    <div className="space-y-32 py-24">
      {/* HOW IT WORKS */}
      <section id="how-it-works" className="container mx-auto px-4 max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.35 }}
          transition={{ type: "spring", stiffness: 240, damping: 22 }}
          className="text-center mb-20"
        >
          <div className="inline-flex items-center gap-2 rounded-full border border-indigo-200 bg-indigo-50 px-4 py-1.5 text-sm font-medium text-indigo-600 mb-6">
            <Brain className="h-4 w-4" />
            Under the Hood
          </div>

          <h2 className="text-4xl font-bold tracking-tight text-slate-900 mb-4">
            How the Technology Works
          </h2>

          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            The system follows a simple deep learning pipeline implemented and tested
            in Google Colab using a CNN model for pneumonia classification plus
            occlusion sensitivity for explainability.
          </p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.25 }}
          className="grid gap-8 md:grid-cols-3"
        >
          {/* CARD 1 */}
          <motion.div
            variants={itemVariants}
            whileHover={{ y: -6 }}
            transition={{ type: "spring", stiffness: 300, damping: 22 }}
          >
            <Card className="h-full border-none bg-white shadow-lg transition-shadow duration-300 hover:shadow-xl">
              <CardHeader className="p-8 pb-0">
                <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-indigo-600 text-white shadow-lg shadow-indigo-500/25">
                  <FileSearch className="h-7 w-7" />
                </div>
                <CardTitle className="text-xl font-semibold">
                  1. Pre-processing
                </CardTitle>
              </CardHeader>
              <CardContent className="p-8 pt-4 text-slate-600 leading-relaxed">
                The chest X-ray is resized to the model’s input shape and normalized
                so the CNN receives consistent pixel ranges.
              </CardContent>
            </Card>
          </motion.div>

          {/* CARD 2 */}
          <motion.div
            variants={itemVariants}
            whileHover={{ y: -6 }}
            transition={{ type: "spring", stiffness: 300, damping: 22 }}
          >
            <Card className="h-full border-none bg-white shadow-lg transition-shadow duration-300 hover:shadow-xl">
              <CardHeader className="p-8 pb-0">
                <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-indigo-600 text-white shadow-lg shadow-indigo-500/25">
                  <Brain className="h-7 w-7" />
                </div>
                <CardTitle className="text-xl font-semibold">
                  2. Deep Learning Analysis
                </CardTitle>
              </CardHeader>
              <CardContent className="p-8 pt-4 text-slate-600 leading-relaxed">
                A CNN model trained on labeled chest radiographs produces a prediction
                score for pneumonia likelihood.
              </CardContent>
            </Card>
          </motion.div>

          {/* CARD 3 */}
          <motion.div
            variants={itemVariants}
            whileHover={{ y: -6 }}
            transition={{ type: "spring", stiffness: 300, damping: 22 }}
          >
            <Card className="h-full border-none bg-white shadow-lg transition-shadow duration-300 hover:shadow-xl">
              <CardHeader className="p-8 pb-0">
                <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-indigo-600 text-white shadow-lg shadow-indigo-500/25">
                  <Lock className="h-7 w-7" />
                </div>
                <CardTitle className="text-xl font-semibold">
                  3. Occlusion Mapping
                </CardTitle>
              </CardHeader>
              <CardContent className="p-8 pt-4 text-slate-600 leading-relaxed">
                We systematically mask regions of the image and re-run inference to
                generate a heatmap showing which areas most influence the prediction.
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>
      </section>

      {/* DISCLAIMER */}
      <section id="disclaimer" className="container mx-auto px-4 max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.25 }}
          transition={{ type: "spring", stiffness: 220, damping: 22 }}
          className="relative overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-xl"
        >
          <div className="absolute top-0 left-0 h-2 w-full bg-gradient-to-r from-indigo-500 via-purple-500 to-indigo-500" />

          <div className="p-10 md:p-12">
            <div className="flex flex-col gap-10 md:flex-row md:items-start">
              <div className="flex-shrink-0">
                <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-indigo-50 text-indigo-600 ring-1 ring-indigo-100">
                  <ShieldAlert className="h-9 w-9" />
                </div>
              </div>

              <div className="flex-1">
                <div className="flex flex-wrap items-center gap-4">
                  <h3 className="text-3xl font-bold tracking-tight text-slate-900">
                    Research Preview
                  </h3>
                  <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-slate-600">
                    NON-CLINICAL
                  </span>
                </div>

                <p className="mt-4 text-slate-600 leading-relaxed text-lg">
                  This AI model is a demonstration of deep learning capabilities in
                  medical imaging and is{" "}
                  <span className="font-semibold text-slate-900 underline decoration-indigo-400 decoration-2 underline-offset-4">
                    not intended for diagnostic use.
                  </span>
                </p>

                <div className="my-8 h-px w-full bg-slate-200/70" />

                <div className="grid gap-10 md:grid-cols-2">
                  <div className="flex gap-5">
                    <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-indigo-50 text-indigo-600 ring-1 ring-indigo-100">
                      <ShieldCheck className="h-6 w-6" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-slate-900">
                        Educational Purpose Only
                      </h4>
                      <p className="mt-1 text-slate-600 leading-relaxed">
                        Designed for research and academic demonstration. This tool
                        has not been validated by FDA or regulatory bodies.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-5">
                    <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-indigo-50 text-indigo-600 ring-1 ring-indigo-100">
                      <Stethoscope className="h-6 w-6" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-slate-900">
                        Consult a Professional
                      </h4>
                      <p className="mt-1 text-slate-600 leading-relaxed">
                        Always seek the advice of a qualified health provider with
                        any questions regarding a medical condition.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </section>
    </div>
  );
}
