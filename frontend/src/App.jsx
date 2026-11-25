import React, { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Youtube,
  UploadCloud,
  CheckCircle2,
  XCircle,
  Send,
  Loader2,
  AlertCircle,
} from "lucide-react";

const API_BASE = "http://localhost:8000/api";

export default function App() {
  const [online, setOnline] = useState(false);
  const [savedVideos, setSavedVideos] = useState({});

  const [videoUrl, setVideoUrl] = useState("");
  const [loadingVideo, setLoadingVideo] = useState(false);
  const [activeVideo, setActiveVideo] = useState(null);
  const [error, setError] = useState("");

  const [tab, setTab] = useState("qa");
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [messages, setMessages] = useState([]);

  const [notesLang, setNotesLang] = useState("English");
  const [loadingNotes, setLoadingNotes] = useState(false);
  const [notes, setNotes] = useState("");

  const [quizLang, setQuizLang] = useState("English");
  const [loadingQuiz, setLoadingQuiz] = useState(false);
  const [quiz, setQuiz] = useState([]);
  const [answers, setAnswers] = useState({});
  const [quizSubmitted, setQuizSubmitted] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then((d) => {
        setOnline(true);
        setSavedVideos(d.saved_videos || {});
      })
      .catch(() => setOnline(false));
  }, []);

  const handleProcessVideo = async (e) => {
    e.preventDefault();
    if (!videoUrl.trim()) return;
    setError("");
    setLoadingVideo(true);
    try {
      const res = await fetch(`${API_BASE}/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: videoUrl.trim() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to process video.");
      setActiveVideo(data);
      setSavedVideos((prev) => ({ ...prev, [data.video_id]: data.title }));
      setMessages([]);
      setNotes("");
      setQuiz([]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingVideo(false);
    }
  };

  const selectSavedVideo = (vid, title) => {
    setActiveVideo({ video_id: vid, title });
    setMessages([]);
    setNotes("");
    setQuiz([]);
    setError("");
  };

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question.trim() || !activeVideo) return;
    const q = question.trim();
    setQuestion("");
    setError("");
    setMessages((prev) => [...prev, { role: "user", text: q }]);
    setAsking(true);
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: activeVideo.video_id,
          question: q,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to get answer.");
      setMessages((prev) => [...prev, { role: "assistant", ...data }]);
    } catch (err) {
      setError(err.message);
    } finally {
      setAsking(false);
    }
  };

  const handleGenerateNotes = async () => {
    if (!activeVideo) return;
    setError("");
    setLoadingNotes(true);
    try {
      const res = await fetch(`${API_BASE}/notes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: activeVideo.video_id,
          language: notesLang,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to generate notes.");
      setNotes(data.notes);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingNotes(false);
    }
  };

  const handleGenerateQuiz = async () => {
    if (!activeVideo) return;
    setError("");
    setLoadingQuiz(true);
    setQuizSubmitted(false);
    setAnswers({});
    try {
      const res = await fetch(`${API_BASE}/quiz`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: activeVideo.video_id,
          language: quizLang,
          num_questions: 5,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to generate quiz.");
      setQuiz(data.quiz || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingQuiz(false);
    }
  };

  return (
    <div className="min-h-screen bg-white text-slate-900 flex flex-col font-sans">
      {/* Centered Header with Red Logo */}
      <header className="relative border-b border-slate-200 px-6 py-4 flex items-center justify-center">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-red-600 text-white flex items-center justify-center shadow-sm">
            <Youtube className="w-5 h-5" />
          </div>
          <h1 className="text-xl font-bold tracking-tight text-slate-900">
            AI Video Learning Suite
          </h1>
        </div>

        {/* Right Engine Status */}
        <div className="hidden sm:flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-slate-200 text-xs font-medium text-slate-700 absolute right-6">
          <span className={`w-2 h-2 rounded-full ${online ? "bg-emerald-500" : "bg-rose-500"}`} />
          <span>{online ? "Engine Online" : "Engine Offline"}</span>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 w-full max-w-7xl mx-auto px-6 py-6 flex flex-col items-center">
        {error && (
          <div className="w-full mb-5 p-3 rounded-xl border border-rose-200 bg-rose-50 text-rose-700 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {!activeVideo ? (
          <div className="w-full max-w-xl my-auto space-y-6 text-center">
            <h2 className="text-xl font-bold">Paste YouTube video link for understanding</h2>
            <form
              onSubmit={handleProcessVideo}
              className="border-2 border-dashed border-slate-300 rounded-3xl p-10 flex flex-col items-center gap-5"
            >
              <div className="w-12 h-12 rounded-2xl bg-red-50 text-red-600 flex items-center justify-center">
                <UploadCloud className="w-6 h-6" />
              </div>

              <input
                type="url"
                required
                value={videoUrl}
                onChange={(e) => setVideoUrl(e.target.value)}
                placeholder="https://www.youtube.com/watch?v=..."
                className="w-full px-4 py-2.5 text-sm border border-slate-300 rounded-xl focus:outline-none focus:border-red-600 text-center"
              />

              <button
                type="submit"
                disabled={loadingVideo}
                className="px-6 py-2.5 rounded-xl bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white text-sm font-semibold flex items-center gap-2"
              >
                {loadingVideo ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Indexing Video...</span>
                  </>
                ) : (
                  <span>Process Video</span>
                )}
              </button>
            </form>

            {Object.keys(savedVideos).length > 0 && (
              <div className="pt-4 text-left space-y-2">
                <div className="text-xs font-bold uppercase tracking-wider text-slate-500 text-center">
                  Or choose a previously indexed video
                </div>
                <div className="flex flex-wrap justify-center gap-2">
                  {Object.entries(savedVideos).map(([vid, title]) => (
                    <button
                      key={vid}
                      onClick={() => selectSavedVideo(vid, title)}
                      className="px-3.5 py-1.5 rounded-full border border-slate-200 hover:border-red-600 hover:text-red-600 text-xs font-medium text-slate-700 transition"
                    >
                      {title}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="w-full space-y-4">
            {/* Top Bar: Active Video Title & Switch Button */}
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 pb-3">
              <h2 className="text-base font-bold text-slate-900">{activeVideo.title}</h2>
              <button
                onClick={() => setActiveVideo(null)}
                className="text-xs font-semibold text-red-600 hover:underline"
              >
                + Index another video
              </button>
            </div>

            {/* Side-by-Side Layout: Video Left, Tools Right */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Left Column: Embedded YouTube Video Player */}
              <div className="lg:col-span-6 lg:sticky lg:top-6">
                <div className="w-full aspect-video rounded-2xl overflow-hidden border border-slate-200 bg-black shadow-sm">
                  <iframe
                    className="w-full h-full"
                    src={`https://www.youtube.com/embed/${activeVideo.video_id}`}
                    title={activeVideo.title}
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                  />
                </div>
              </div>

              {/* Right Column: Tabs + Q&A / Quiz / Study Notes */}
              <div className="lg:col-span-6 border border-slate-200 rounded-2xl p-4 bg-white shadow-sm space-y-4">
                {/* Tabs */}
                <div className="border-b border-slate-200 pb-3 flex items-center gap-2">
                  {[
                    { id: "qa", label: "Q&A Assistant" },
                    { id: "quiz", label: "Quiz" },
                    { id: "notes", label: "Study Notes" },
                  ].map((t) => (
                    <button
                      key={t.id}
                      onClick={() => setTab(t.id)}
                      className={`flex-1 py-2 rounded-xl text-xs font-semibold transition ${
                        tab === t.id
                          ? "bg-red-600 text-white shadow-sm"
                          : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                      }`}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>

                {/* TAB 1: Q&A */}
                {tab === "qa" && (
                  <div className="space-y-4">
                    <div className="space-y-3 h-[380px] overflow-y-auto pr-1">
                      {messages.length === 0 && (
                        <div className="h-full flex items-center justify-center text-xs text-slate-400 text-center px-6">
                          Ask any question about this video in English or Roman Urdu.
                        </div>
                      )}
                      {messages.map((m, i) =>
                        m.role === "user" ? (
                          <div key={i} className="flex justify-end">
                            <div className="bg-slate-900 text-white text-sm px-4 py-2.5 rounded-2xl max-w-md">
                              {m.text}
                            </div>
                          </div>
                        ) : (
                          <div key={i} className="border border-slate-200 rounded-2xl p-4 space-y-2.5 text-sm">
                            <div className="flex items-center justify-between text-xs">
                              {m.verified ? (
                                <span className="inline-flex items-center gap-1 font-semibold text-emerald-700">
                                  <CheckCircle2 className="w-3.5 h-3.5" />
                                  {m.mode} ({m.confidence}%)
                                </span>
                              ) : (
                                <span className="inline-flex items-center gap-1 font-semibold text-amber-700">
                                  <XCircle className="w-3.5 h-3.5" />
                                  {m.mode}
                                </span>
                              )}
                              {m.latency_ms > 0 && <span className="text-slate-400">{m.latency_ms} ms</span>}
                            </div>

                            <p className="text-slate-800 leading-relaxed">{m.answer}</p>

                            {m.quote && (
                              <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-600">
                                <strong>Proof:</strong> "{m.quote}"
                              </div>
                            )}

                            {m.verified && (
                              <a
                                href={`https://www.youtube.com/watch?v=${activeVideo.video_id}&t=${m.timestamp}s`}
                                target="_blank"
                                rel="noreferrer"
                                className="inline-block text-xs font-semibold text-red-600 underline"
                              >
                                Watch from {m.timestamp}s →
                              </a>
                            )}
                          </div>
                        )
                      )}
                    </div>

                    <form onSubmit={handleAsk} className="flex gap-2 pt-2 border-t border-slate-100">
                      <input
                        type="text"
                        required
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        placeholder="Ask in English or Roman Urdu..."
                        className="flex-1 px-4 py-2.5 text-sm border border-slate-300 rounded-xl focus:outline-none focus:border-red-600"
                      />
                      <button
                        type="submit"
                        disabled={asking}
                        className="px-5 py-2.5 rounded-xl bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white text-xs font-semibold flex items-center gap-1.5"
                      >
                        {asking ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                        <span>Ask</span>
                      </button>
                    </form>
                  </div>
                )}

                {/* TAB 2: QUIZ */}
                {tab === "quiz" && (
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <select
                        value={quizLang}
                        onChange={(e) => setQuizLang(e.target.value)}
                        className="px-3 py-2 text-xs border border-slate-300 rounded-lg"
                      >
                        <option>English</option>
                        <option>Roman Urdu</option>
                      </select>
                      <button
                        onClick={handleGenerateQuiz}
                        disabled={loadingQuiz}
                        className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white text-xs font-semibold flex items-center gap-1.5"
                      >
                        {loadingQuiz && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                        <span>Generate Quiz</span>
                      </button>
                    </div>

                    <div className="space-y-3 max-h-[420px] overflow-y-auto pr-1">
                      {quiz.map((item, idx) => (
                        <div key={idx} className="p-4 border border-slate-200 rounded-xl space-y-2 text-xs">
                          <div className="font-bold text-sm">
                            {idx + 1}. {item.question}
                          </div>
                          <div className="space-y-1.5">
                            {item.options.map((opt, oIdx) => {
                              const chosen = answers[idx] === opt;
                              let cls = "border-slate-200 hover:bg-slate-50";
                              if (quizSubmitted) {
                                if (opt === item.answer) cls = "border-emerald-500 bg-emerald-50 text-emerald-900";
                                else if (chosen) cls = "border-rose-500 bg-rose-50 text-rose-900";
                              } else if (chosen) {
                                cls = "border-slate-900 bg-slate-100";
                              }
                              return (
                                <button
                                  key={oIdx}
                                  onClick={() => !quizSubmitted && setAnswers({ ...answers, [idx]: opt })}
                                  className={`w-full text-left px-3 py-2 rounded-lg border ${cls}`}
                                >
                                  {opt}
                                </button>
                              );
                            })}
                          </div>
                          {quizSubmitted && <p className="text-slate-600 pt-1">{item.explanation}</p>}
                        </div>
                      ))}
                    </div>

                    {quiz.length > 0 && (
                      <button
                        onClick={() => setQuizSubmitted(true)}
                        className="px-5 py-2 rounded-xl bg-red-600 hover:bg-red-700 text-white text-xs font-semibold"
                      >
                        Submit Quiz
                      </button>
                    )}
                  </div>
                )}

                {/* TAB 3: STUDY NOTES */}
                {tab === "notes" && (
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <select
                        value={notesLang}
                        onChange={(e) => setNotesLang(e.target.value)}
                        className="px-3 py-2 text-xs border border-slate-300 rounded-lg"
                      >
                        <option>English</option>
                        <option>Roman Urdu</option>
                      </select>
                      <button
                        onClick={handleGenerateNotes}
                        disabled={loadingNotes}
                        className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white text-xs font-semibold flex items-center gap-1.5"
                      >
                        {loadingNotes && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                        <span>Generate Notes</span>
                      </button>
                    </div>

                    {notes && (
                      <div className="max-h-[420px] overflow-y-auto p-4 border border-slate-200 rounded-xl bg-white text-sm text-slate-800 leading-relaxed space-y-3 [&_h1]:text-base [&_h1]:font-bold [&_h2]:text-sm [&_h2]:font-bold [&_h2]:mt-3 [&_h3]:text-xs [&_h3]:font-bold [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5 [&_table]:w-full [&_table]:border-collapse [&_th]:border [&_th]:border-slate-300 [&_th]:bg-slate-50 [&_th]:px-2 [&_th]:py-1.5 [&_th]:text-left [&_td]:border [&_td]:border-slate-300 [&_td]:px-2 [&_td]:py-1.5 [&_hr]:border-slate-200">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {notes.replace(/<br\s*\/?>/gi, "\n")}
                        </ReactMarkdown>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}