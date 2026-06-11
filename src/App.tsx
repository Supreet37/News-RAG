import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  BookOpen,
  Sparkles,
  Cpu,
  Database,
  Search,
  Bot,
  FileText,
  Play,
  CheckCircle2,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Activity,
  Info,
  Server,
  ArrowRight,
  TrendingUp,
  Sliders,
  HelpCircle,
  Download
} from "lucide-react";
import { jsPDF } from "jspdf";
import { SAMPLES } from "./data";
import { Chunk, PipelineLog } from "./types";

// ── TF-IDF cosine-similarity client-side retrieval engine ──
const STOPWORDS = new Set([
  "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
  "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
  "did", "will", "would", "could", "should", "this", "that", "it", "its", "they",
  "we", "you", "he", "she", "as", "by", "from", "so", "if", "than", "after", "before"
]);

function tokenize(text: string): string[] {
  return text.toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(w => w.length > 2 && !STOPWORDS.has(w));
}

function tfidf(term: string, tokens: string[], allTokenArrays: string[][]): number {
  const tf = tokens.filter(t => t === term).length / tokens.length;
  const df = allTokenArrays.filter(arr => arr.includes(term)).length;
  const idf = df ? Math.log(allTokenArrays.length / df) : 0;
  return tf * idf;
}

function cosineSim(qTokens: string[], cTokens: string[], allTokenArrays: string[][]): number {
  const vocab = [...new Set([...qTokens, ...cTokens])];
  const qVec = vocab.map(t => tfidf(t, qTokens, allTokenArrays));
  const cVec = vocab.map(t => tfidf(t, cTokens, allTokenArrays));
  const dot = qVec.reduce((sum, val, idx) => sum + val * cVec[idx], 0);
  const magnitude = (vec: number[]) => Math.sqrt(vec.reduce((sum, x) => sum + x * x, 0));
  const magQ = magnitude(qVec);
  const magC = magnitude(cVec);
  return (magQ && magC) ? dot / (magQ * magC) : 0;
}

function retrieveTopChunks(question: string, allChunks: string[], topK = 3): { idx: number; text: string; score: number }[] {
  const qTokens = tokenize(question);
  const allTokenArrays = allChunks.map(c => tokenize(c));
  return allChunks
    .map((chunk, i) => ({
      idx: i,
      text: chunk,
      score: cosineSim(qTokens, allTokenArrays[i], allTokenArrays)
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .filter(r => r.score > 0);
}

// ── Chunking function ──
function chunkText(text: string, chunkSize = 150, overlap = 25): string[] {
  const sentences = text.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 10);
  const result: string[] = [];
  let current: string[] = [];
  let currentLen = 0;

  for (const sentence of sentences) {
    const words = sentence.split(/\s+/);
    if (currentLen + words.length > chunkSize && current.length > 0) {
      result.push(current.join(" "));
      const overlapWords = current.join(" ").split(/\s+/).slice(-overlap);
      current = [...overlapWords, ...words];
      currentLen = current.length;
    } else {
      current.push(...words);
      currentLen += words.length;
    }
  }
  if (current.length > 0) {
    result.push(current.join(" "));
  }
  return result.filter(c => c.trim().length > 30);
}

export default function App() {
  // Active states
  const [docInput, setDocInput] = useState<string>(SAMPLES[0].text);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [answer, setAnswer] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [selectedSampleKey, setSelectedSampleKey] = useState<string>("ai");
  const [pipelineLogs, setPipelineLogs] = useState<PipelineLog[]>([]);
  const [showLogs, setShowLogs] = useState<boolean>(false);
  const [retrievedChunkIdxs, setRetrievedChunkIdxs] = useState<number[]>([]);
  const [errorMessage, setErrorMessage] = useState<string>("");

  // Server health state
  const [serverStatus, setServerStatus] = useState<"loading" | "ok" | "error">("loading");
  const [serverMessage, setServerMessage] = useState<string>("Initializing server connection...");

  // Metrics
  const [wordCount, setWordCount] = useState<number>(0);
  const [approxTokens, setApproxTokens] = useState<number>(0);

  // References
  const chunksGridRef = useRef<HTMLDivElement>(null);
  const answerSectionRef = useRef<HTMLDivElement>(null);

  // Check backend server connection
  useEffect(() => {
    checkServerHealth();
  }, []);

  // Sync metrics on text input update
  useEffect(() => {
    const cleanText = docInput.trim();
    if (!cleanText) {
      setWordCount(0);
      setApproxTokens(0);
      return;
    }
    const words = cleanText.split(/\s+/).length;
    setWordCount(words);
    setApproxTokens(Math.round(words * 1.3));
  }, [docInput]);

  // Initial auto-chunking
  useEffect(() => {
    handleProcessDocument(true);
  }, []);

  const checkServerHealth = async () => {
    setServerStatus("loading");
    setServerMessage("Connecting of application server...");
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.keySet) {
        setServerStatus("ok");
        setServerMessage("✓ Groq LLaMA 3.3 Connected · Ready for indexing");
      } else {
        setServerStatus("error");
        setServerMessage("⚠ Groq API key is not configured.");
      }
    } catch (err) {
      setServerStatus("error");
      setServerMessage("✗ Backend server unreachable. Make sure the container is healthy.");
    }
  };

  const handleLoadSample = (key: string) => {
    const sample = SAMPLES.find(s => s.key === key);
    if (sample) {
      setSelectedSampleKey(key);
      setDocInput(sample.text);
      setAnswer("");
      setQuestion("");
      setRetrievedChunkIdxs([]);
      setPipelineLogs([]);
      // Auto process loaded sample
      setTimeout(() => {
        const rawChunks = chunkText(sample.text);
        const mappedChunks: Chunk[] = rawChunks.map((c, i) => ({ idx: i, text: c }));
        setChunks(mappedChunks);
      }, 50);
    }
  };

  const handleProcessDocument = (silent = false) => {
    const text = docInput.trim();
    if (!text) {
      if (!silent) setErrorMessage("Please paste or type some document text first.");
      return;
    }

    const rawChunks = chunkText(text);
    if (rawChunks.length === 0) {
      if (!silent) setErrorMessage("Document is too short to split into meaningful chunks.");
      return;
    }

    const mappedChunks: Chunk[] = rawChunks.map((c, i) => ({ idx: i, text: c }));
    setChunks(mappedChunks);
    setRetrievedChunkIdxs([]);
    setErrorMessage("");

    if (!silent) {
      // Small feedback pop
      const alertDiv = document.createElement("div");
      alertDiv.className = "fixed bottom-5 right-5 bg-emerald-500 text-white px-4 py-2 rounded-lg shadow-xl text-sm font-medium z-50 flex items-center gap-2";
      alertDiv.innerHTML = `<span>⚡ Chrono Indexed ${mappedChunks.length} chunks successfully!</span>`;
      document.body.appendChild(alertDiv);
      setTimeout(() => alertDiv.remove(), 2500);
    }
  };

  const handleAskQuestion = async (selectedQuestion?: string) => {
    const q = (selectedQuestion || question).trim();
    setErrorMessage("");

    if (chunks.length === 0) {
      setErrorMessage("⚠️ Please process & index a document first before asking questions.");
      return;
    }

    if (!q) {
      setErrorMessage("⚠️ Please write a question.");
      return;
    }

    // Set interactive question field
    if (selectedQuestion) {
      setQuestion(selectedQuestion);
    }

    setIsLoading(true);
    setAnswer("");
    const temporaryLogs: PipelineLog[] = [];

    // Step 1: Retrieval
    temporaryLogs.push({
      icon: "🔍",
      title: "Index Scan & Term TF-IDF Matching",
      content: `Scanning ${chunks.length} local vector index slots. Mapping question terms: [${tokenize(q).join(", ")}] against local collection`
    });
    setPipelineLogs([...temporaryLogs]);

    await new Promise((r) => setTimeout(r, 400));

    // Calculate similarity locally
    const retrieved = retrieveTopChunks(q, chunks.map(c => c.text), 3);

    if (retrieved.length === 0) {
      setErrorMessage("Could not identify high-similarity chunks in this document. Please test with another prompt.");
      setIsLoading(false);
      return;
    }

    // Highlight chunks in the screen
    const retrievedIndices = retrieved.map(r => r.idx);
    setRetrievedChunkIdxs(retrievedIndices);

    // Document feedback logging
    temporaryLogs.push({
      icon: "✅",
      title: "Top Chunks Retrieved",
      content: `Ranked top ${retrieved.length} chunks. Selected indexes: ${retrievedIndices.map(i => `#${i + 1}`).join(", ")} | Cross-correlation similarity scores: ${retrieved.map(r => r.score.toFixed(3)).join(", ")}`
    });
    setPipelineLogs([...temporaryLogs]);

    await new Promise((r) => setTimeout(r, 300));

    // Step 2: Send to backend
    temporaryLogs.push({
      icon: "🔒",
      title: "Secure API Query Formulation",
      content: "Forwarding retrieved document chunks and user prompts to the Groq LLaMA 3.3 server-side middleware."
    });
    setPipelineLogs([...temporaryLogs]);

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          chunks: retrieved
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `Server query returned state ${response.status}`);
      }

      temporaryLogs.push({
        icon: "✨",
        title: "LLaMA Generates Context Response",
        content: `Prompt synthesis successful. Model context evaluated fully using active backend pipeline.`
      });
      setPipelineLogs([...temporaryLogs]);

      setAnswer(data.answer);

      // Smooth scroll to answer
      setTimeout(() => {
        answerSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }, 100);

    } catch (err: any) {
      setErrorMessage(err.message || "An unexpected error occurred contacting the RAG server.");
    } finally {
      setIsLoading(false);
    }
  };

  const getDocSuggestions = () => {
    if (selectedSampleKey === "ai") {
      return [
        "What is agentic AI and what companies launched it?",
        "How much was India's budget for the IndiaAI Mission?",
        "What is India's role in the global AI ecosystem?",
        "What is the EU AI Act?"
      ];
    } else if (selectedSampleKey === "market") {
      return [
        "Which companies form the Magnificent Seven, and what's their percentage load in S&P 500?",
        "What caused NVIDIA's market cap to exceed $3 trillion?",
        "What is Sensex's current performance and India's base GDP growth?",
        "Why did the cryptocurrency market surge in late 2024?"
      ];
    } else {
      return [
        "How do wearables help predict cardiac events?",
        "What is Ayushman Bharat Digital Mission and its registered user base?",
        "How is generative AI accelerating drug discovery?",
        "What are some popular mental health tech apps?"
      ];
    }
  };

  const handleExportPDF = () => {
    if (!answer) return;

    const doc = new jsPDF();

    // Title / Header Styling
    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    doc.setTextColor(30, 41, 59); // slate-800
    doc.text("RAG Cognitive Assistant Search Report", 15, 22);

    // Decorative baseline rule
    doc.setDrawColor(79, 70, 229); // indigo-600 tint
    doc.setLineWidth(1);
    doc.line(15, 27, 195, 27);

    // Metadata section
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(100, 116, 139); // slate-500
    const timeGenerated = new Date().toLocaleString("en-US", {
      dateStyle: "long",
      timeStyle: "medium"
    });
    doc.text(`Generated: ${timeGenerated}`, 15, 33);
    doc.text("Source Engine: Groq LLaMA 3.3 Versatile (70B Model Inference)", 15, 38);

    doc.setDrawColor(226, 232, 240); // slate-200
    doc.setLineWidth(0.5);
    doc.line(15, 42, 195, 42);

    let startY = 49;

    // Optional Question Section
    if (question) {
      doc.setFont("helvetica", "bold");
      doc.setFontSize(11);
      doc.setTextColor(79, 70, 229); // indigo-600
      doc.text("Inquired Query / Question:", 15, startY);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(10.5);
      doc.setTextColor(15, 23, 42); // slate-900
      const splitQuestion = doc.splitTextToSize(`"${question}"`, 180);
      doc.text(splitQuestion, 15, startY + 6);

      const questionHeight = splitQuestion.length * 5.5;
      startY = startY + 6 + questionHeight + 8;
    }

    // Synthesized Reply heading
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(79, 70, 229); // indigo-600
    doc.text("Synthesized Reply:", 15, startY);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(51, 65, 85); // slate-700

    const splitAnswer = doc.splitTextToSize(answer, 180);
    let currentY = startY + 7;
    const pageHeight = doc.internal.pageSize.height;
    const bottomMargin = 20;

    for (let i = 0; i < splitAnswer.length; i++) {
      if (currentY > pageHeight - bottomMargin) {
        doc.addPage();
        currentY = 22; // reset for new page
      }
      doc.text(splitAnswer[i], 15, currentY);
      currentY += 5.5;
    }

    // Footer at the bottom of the last page
    if (currentY < pageHeight - 15) {
      doc.setFont("helvetica", "italic");
      doc.setFontSize(8);
      doc.setTextColor(148, 163, 184); // slate-400
      doc.text("Report generated dynamically on modern sandboxed RAG edge storage.", 15, pageHeight - 10);
    }

    doc.save("rag-synthesized-response.pdf");
  };

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-900 font-sans selection:bg-indigo-600/10 selection:text-indigo-900">
      
      {/* ── Outer Background & Header ── */}
      <header className="border-b border-slate-200/80 bg-white/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-indigo-600 text-white rounded-xl shadow-md shadow-indigo-100 flex items-center justify-center">
              <BookOpen className="w-4 h-4" />
            </div>
            <div>
              <h1 className="text-lg font-extrabold tracking-tight text-slate-900">
                Cognitive RAG Assistant
              </h1>
              <p className="text-[10.5px] text-slate-400 font-medium tracking-wide">
                Fast Inference Engine powered by Groq LLaMA 3.3
              </p>
            </div>
          </div>

          {/* Beautiful Active Pipeline Badges representing real progress */}
          <div className="flex flex-wrap items-center gap-1.5 md:gap-2.5">
            <span className={`px-2.5 py-1 rounded-full text-[10.5px] font-medium border transition-all ${chunks.length > 0 ? "bg-indigo-50 border-indigo-100 text-indigo-700" : "bg-slate-100 border-slate-200 text-slate-400"}`}>
              ① Ingested
            </span>
            <span className="text-slate-300 text-xs font-light">→</span>
            <span className={`px-2.5 py-1 rounded-full text-[10.5px] font-medium border transition-all ${chunks.length > 0 ? "bg-cyan-50 border-cyan-100 text-cyan-700" : "bg-slate-100 border-slate-200 text-slate-400"}`}>
              ② Chunked
            </span>
            <span className="text-slate-300 text-xs font-light">→</span>
            <span className={`px-2.5 py-1 rounded-full text-[10.5px] font-medium border transition-all ${retrievedChunkIdxs.length > 0 ? "bg-emerald-50 border-emerald-100 text-emerald-700" : "bg-slate-100 border-slate-200 text-slate-400"}`}>
              ③ Retrieved
            </span>
            <span className="text-slate-300 text-xs font-light">→</span>
            <span className={`px-2.5 py-1 rounded-full text-[10.5px] font-medium border transition-all ${answer ? "bg-purple-50 border-purple-100 text-purple-700 font-semibold shadow-sm" : "bg-slate-100 border-slate-200 text-slate-400"}`}>
              ④ Response Ready
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        
        {/* ── Server Status Alert Bar ── */}
        {/* <div className={`p-4 rounded-xl border flex items-center justify-between transition-all shadow-sm ${serverStatus === "ok" ? "bg-emerald-50/50 border-emerald-200 text-emerald-900" : serverStatus === "loading" ? "bg-amber-50/50 border-amber-200 text-amber-900 animate-pulse" : "bg-rose-50/50 border-rose-200 text-rose-900"}`}>
          <div className="flex items-center gap-2.5 text-xs font-medium">
            <span className={`w-2.5 h-2.5 rounded-full ${serverStatus === "ok" ? "bg-emerald-600 animate-pulse" : serverStatus === "loading" ? "bg-amber-500 animate-ping" : "bg-rose-600"}`} />
            <span>{serverMessage}</span>
          </div>
          {serverStatus !== "ok" && (
            <button 
              onClick={checkServerHealth}
              className="text-xs font-bold uppercase tracking-wider text-indigo-600 hover:text-indigo-700 hover:underline flex items-center gap-1.5 bg-white px-3 py-1.5 rounded-lg border border-slate-200"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Reconnect
            </button>
          )}
        </div> */}

        {/* ── Main Layout Workspace Split Screen ── */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Block: Document Formulation Input */}
          <div className="lg:col-span-7 space-y-6">
            <div className="bg-white border border-slate-200/80 rounded-2xl p-6 space-y-6 shadow-sm">
              <div className="flex items-center justify-between pb-4 border-b border-slate-100">
                <div className="flex items-center gap-2.5">
                  <div className="w-7 h-7 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center text-xs font-bold">1</div>
                  <h2 className="text-base font-bold text-slate-900">
                    Source Knowledge Document
                  </h2>
                </div>
                <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 bg-slate-50 px-3 py-1 rounded-lg border border-slate-100">
                  <span>{wordCount} words</span>
                  <span className="text-slate-300">|</span>
                  <span>~{approxTokens} tokens</span>
                </div>
              </div>

              {/* Sample Selector Row */}
              <div className="space-y-2">
                <label className="text-xs font-bold text-slate-500 uppercase tracking-wider block">
                  Select sample article:
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                  {SAMPLES.map(sample => (
                    <button
                      key={sample.key}
                      onClick={() => handleLoadSample(sample.key)}
                      className={`py-2.5 px-4 text-left border text-xs font-semibold transition-all flex items-center justify-between cursor-pointer rounded-xl ${selectedSampleKey === sample.key ? "bg-indigo-600 border-indigo-600 text-white shadow-md shadow-indigo-100" : "bg-slate-50 hover:bg-slate-100 border-slate-200 text-slate-600"}`}
                    >
                      <span className="truncate">{sample.title}</span>
                      <span className="text-base ml-1 opacity-80">{sample.icon}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Document Textarea */}
              <div className="space-y-2">
                <label className="text-xs font-bold text-slate-500 uppercase tracking-wider block">
                  Custom document input:
                </label>
                <textarea
                  value={docInput}
                  onChange={(e) => {
                    setDocInput(e.target.value);
                    setSelectedSampleKey(""); // Reset sample highlighting when editing manually
                  }}
                  className="w-full bg-slate-50 hover:bg-slate-50/50 focus:bg-white border border-slate-200 rounded-xl p-4 text-sm text-slate-800 leading-relaxed placeholder-slate-400 focus:outline-none focus:ring-4 focus:ring-indigo-50 focus:border-indigo-500 transition-all h-[260px] resize-y shadow-inner"
                  placeholder="Paste or write your document content here (minimum length > 50 words to process)..."
                />
              </div>

              {/* Process Buttons */}
              <div className="pt-2">
                <button
                  onClick={() => handleProcessDocument()}
                  className="w-full h-11 rounded-xl bg-indigo-600 text-white text-xs font-bold tracking-wider uppercase hover:bg-indigo-700 transition-all flex items-center justify-center gap-2 cursor-pointer shadow-md shadow-indigo-100 active:scale-[0.99] duration-150"
                >
                  <RefreshCw className="w-3.5 h-3.5 animate-spin-reverse" />
                  Index &amp; Chunk Text Document
                </button>
              </div>
            </div>

            {/* Document stats */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-white border border-slate-200/80 rounded-2xl p-5 text-center shadow-sm hover:shadow-md transition-shadow">
                <div className="text-2xl sm:text-3xl font-extrabold text-indigo-600 tracking-tight">{chunks.length}</div>
                <div className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mt-1">Parts Chunks</div>
              </div>
              <div className="bg-white border border-slate-200/80 rounded-2xl p-5 text-center shadow-sm hover:shadow-md transition-shadow">
                <div className="text-2xl sm:text-3xl font-extrabold text-indigo-600 tracking-tight">{wordCount}</div>
                <div className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mt-1">Total Words</div>
              </div>
              <div className="bg-white border border-slate-200/80 rounded-2xl p-5 text-center shadow-sm hover:shadow-md transition-shadow">
                <div className="text-2xl sm:text-3xl font-extrabold text-indigo-600 tracking-tight">{~~approxTokens}</div>
                <div className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mt-1">Est Tokens</div>
              </div>
            </div>
          </div>

          {/* Right Column: Interactive Knowledge Base Index Lists */}
          <div className="lg:col-span-5 space-y-4 flex flex-col h-full self-stretch">
            <div className="bg-white border border-slate-200/80 rounded-2xl p-6 flex flex-col h-full self-stretch min-h-[480px] shadow-sm">
              
              <div className="flex items-center justify-between pb-4 border-b border-slate-100 mb-4">
                <div className="flex items-center gap-2.5">
                  <div className="w-7 h-7 rounded-lg bg-cyan-50 text-cyan-600 flex items-center justify-center text-xs font-bold">2</div>
                  <h2 className="text-base font-bold text-slate-900">
                    Knowledge Index ({chunks.length})
                  </h2>
                </div>
                <span className="text-[10px] font-bold text-cyan-600 bg-cyan-50 px-2.5 py-0.5 rounded-full border border-cyan-100 uppercase tracking-wider">
                  Active Stores
                </span>
              </div>

              {/* Scrollable list containing mapped indexed text blocks */}
              <div 
                ref={chunksGridRef}
                className="flex-1 overflow-y-auto space-y-3.5 pr-1 max-h-[410px] scroll-smooth"
              >
                {chunks.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-center p-8 space-y-3 opacity-60">
                    <Database className="w-8 h-8 text-slate-400" />
                    <div>
                      <p className="text-slate-700 text-sm font-semibold">Memory Register Empty</p>
                      <p className="text-slate-400 text-xs mt-1 pr-4 pl-4 leading-relaxed">
                        Insert a knowledge base document on the left, then trigger indexing to allocate retrieval blocks.
                      </p>
                    </div>
                  </div>
                ) : (
                  chunks.map((chunk) => {
                    const isRetrieved = retrievedChunkIdxs.includes(chunk.idx);
                    return (
                      <motion.div
                        key={chunk.idx}
                        layoutId={`chunk-card-${chunk.idx}`}
                        className={`group border rounded-xl p-4 text-xs transition-all duration-300 ${isRetrieved ? "border-emerald-500 bg-emerald-50/65 text-slate-900 shadow-md shadow-emerald-50" : "border-slate-150 bg-white hover:border-slate-300 text-slate-600"}`}
                      >
                        <div className="flex items-center justify-between mb-2 pb-1.5 border-b border-slate-100/60">
                          <span className={`text-[9.5px] uppercase tracking-wider font-bold ${isRetrieved ? "text-emerald-700" : "text-slate-400"}`}>
                            Memory Block #{String(chunk.idx + 1).padStart(2, "0")}
                          </span>
                          {isRetrieved && (
                            <span className="text-[9.5px] font-extrabold tracking-wider text-emerald-800 uppercase flex items-center gap-1">
                              <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 animate-pulse" />
                              Active Source
                            </span>
                          )}
                        </div>
                        <p className={`leading-relaxed text-xs ${isRetrieved ? "text-slate-800 font-medium" : "text-slate-600"}`}>
                          {chunk.text}
                        </p>
                      </motion.div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        </div>

        {/* ── Ask RAG Question Panel ── */}
        <div className="bg-white border border-slate-200/80 rounded-2xl p-6 space-y-6 shadow-sm">
          <div className="flex items-center gap-2.5 pb-4 border-b border-slate-100">
            <div className="w-7 h-7 rounded-lg bg-purple-50 text-purple-600 flex items-center justify-center text-xs font-bold">3</div>
            <h2 className="text-base font-bold text-slate-900">
              Retrieval Dialog &amp; Reasoning
            </h2>
          </div>

          <div className="space-y-5">
            <div className="relative flex">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !isLoading) {
                    handleAskQuestion();
                  }
                }}
                disabled={isLoading}
                placeholder="Ask a question about the document context..."
                className="flex-1 bg-slate-50 border border-slate-200 focus:bg-white h-12 pl-4 pr-32 text-sm text-slate-800 focus:outline-none focus:ring-4 focus:ring-indigo-50 focus:border-indigo-500 transition-all rounded-xl shadow-inner disabled:opacity-50"
              />
              <button
                onClick={() => handleAskQuestion()}
                disabled={isLoading || !question.trim()}
                className="absolute right-1.5 top-1.5 h-9 px-5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-100 disabled:text-slate-400 hover:text-white text-white text-xs font-bold uppercase transition-all cursor-pointer flex items-center gap-1.5 rounded-lg active:scale-95 duration-100"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    SEARCHING
                  </>
                ) : (
                  <>
                    ASK INFERENCE <ArrowRight className="w-3.5 h-3.5" />
                  </>
                )}
              </button>
            </div>

            {/* Error messaging */}
            {errorMessage && (
              <div className="p-3.5 border border-rose-200 bg-rose-50 text-rose-800 text-xs rounded-xl flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-rose-500 shrink-0" />
                <span>{errorMessage}</span>
              </div>
            )}

            {/* Suggested Chips mapped from selected document */}
            <div className="space-y-2">
              <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                💡 Suggested Queries (Adaptive Context):
              </div>
              <div className="flex flex-wrap gap-2">
                {getDocSuggestions().map((sug, i) => (
                  <button
                    key={i}
                    onClick={() => handleAskQuestion(sug)}
                    disabled={isLoading}
                    className="text-xs font-medium py-1.5 px-3 border border-slate-200 hover:border-indigo-400 text-slate-600 hover:text-indigo-600 bg-white hover:bg-indigo-50/40 transition cursor-pointer text-left rounded-lg duration-150"
                  >
                    {sug}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* ── Generated Answer Display ── */}
        <div ref={answerSectionRef}>
          <AnimatePresence>
            {isLoading && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="bg-white border border-slate-200 rounded-2xl p-8 flex flex-col items-center justify-center py-16 space-y-3 shadow-sm"
              >
                <div className="w-8 h-8 border-3 border-indigo-600/20 border-t-indigo-600 rounded-full animate-spin"></div>
                <p className="text-xs font-bold tracking-widest text-indigo-600 uppercase">
                  Reasoning on provided sources and generating answer...
                </p>
              </motion.div>
            )}

            {answer && !isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white border border-indigo-150 rounded-2xl p-8 space-y-6 shadow-md shadow-indigo-100/15"
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-indigo-50">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg bg-indigo-50 flex items-center justify-center text-indigo-600 font-bold">★</div>
                    <span className="text-sm font-bold text-slate-800 uppercase tracking-wider">
                      Synthesized Reply
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleExportPDF}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-indigo-600 bg-white hover:bg-slate-50 border border-slate-200 hover:border-indigo-300 rounded-lg transition duration-150 cursor-pointer shadow-sm active:scale-95"
                      title="Export this answer as a PDF file"
                    >
                      <Download className="w-3.5 h-3.5 text-indigo-600" />
                      <span>Export PDF</span>
                    </button>
                    <span className="text-[10.5px] font-mono font-bold px-2.5 py-1 rounded-md bg-indigo-50 text-indigo-700 uppercase tracking-wider">
                      LLaMA 3.3 Versatile
                    </span>
                  </div>
                </div>

                {/* Main Content Answer Block */}
                <div className="text-slate-800 text-base leading-relaxed bg-slate-50 p-6 border-l-4 border-indigo-600 rounded-r-xl whitespace-pre-wrap font-sans">
                  {answer}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

      </main>

      {/* Footer message */}
      <footer className="border-t border-slate-200 bg-slate-100/50 text-slate-400 text-center py-6 mt-12">
        <p className="text-[11px] font-medium max-w-2xl mx-auto px-4 leading-relaxed">
          RAG Pipeline Model: TF-IDF local text vectoring with Cosine Similarity math for client-side matching · Groq LLaMA 3.3 Versatile context orchestration.
        </p>
      </footer>
    </div>
  );
}
