"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Msg = { role: "user" | "assistant"; content: string };
type Phase = "idle" | "waking" | "ready";

const LEGACY_STORE_KEY = "morris.chat.v1";
const POLL_MS = 5000;
const WAKE_DEADLINE_MS = 15 * 60 * 1000;

const GREETING: Msg = {
  role: "assistant",
  content:
    "Good day to you. You find me at my work, and glad of the interruption. Ask me what you will: of pattern and craft, of books, or of the condition of England.",
};

export default function MorrisChat() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [draft, setDraft] = useState("");
  const [typing, setTyping] = useState(false);
  const [wakeSeconds, setWakeSeconds] = useState(0);
  const [notice, setNotice] = useState<string | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const aliveRef = useRef(true);

  useEffect(() => {
    aliveRef.current = true;
    try {
      localStorage.removeItem(LEGACY_STORE_KEY);
    } catch {
      // Storage may be unavailable; the in-memory conversation still works.
    }
    return () => {
      aliveRef.current = false;
    };
  }, []);

  useEffect(() => {
    const el = bodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, typing, phase]);

  /** Polls /api/wake until the GPU answers. Resolves true when ready. */
  const pollUntilReady = useCallback(async () => {
    const start = Date.now();
    for (;;) {
      if (!aliveRef.current) return false;
      try {
        const res = await fetch("/api/wake", { cache: "no-store" });
        const data = await res.json();
        if (data.ready) return true;
      } catch {
        /* network hiccup: keep polling */
      }
      if (Date.now() - start > WAKE_DEADLINE_MS) return false;
      setWakeSeconds(Math.round((Date.now() - start) / 1000));
      await new Promise((r) => setTimeout(r, POLL_MS));
    }
  }, []);

  const begin = useCallback(async () => {
    setNotice(null);
    setPhase("waking");
    setWakeSeconds(0);
    const ready = await pollUntilReady();
    if (!aliveRef.current) return;
    if (!ready) {
      setPhase("idle");
      setNotice("He could not be woken. Try again in a little while.");
      return;
    }
    setMessages((prev) => {
      return prev.length === 0 ? [GREETING] : prev;
    });
    setPhase("ready");
    requestAnimationFrame(() => inputRef.current?.focus());
  }, [pollUntilReady]);

  const send = useCallback(async () => {
    const text = draft.trim();
    if (!text || typing) return;
    setDraft("");
    setNotice(null);
    const history: Msg[] = [
      ...messages.filter((message) => message.content.trim().length > 0),
      { role: "user", content: text },
    ];
    setMessages(history);
    setTyping(true);
    try {
      let res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history }),
      });
      if (res.status === 503 || res.status === 504) {
        // The server scaled down again: wake it and retry once.
        setNotice("He stepped away; waking him again.");
        const ready = await pollUntilReady();
        if (!aliveRef.current) return;
        setNotice(null);
        if (ready) {
          res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ messages: history }),
          });
        }
      }
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();
      if (typeof data.content !== "string" || !data.content.trim()) throw new Error("bad reply");
      setMessages((prev) => {
        return [...prev, { role: "assistant", content: data.content }];
      });
    } catch {
      if (aliveRef.current) setNotice("The reply was lost on the way. Send your message again.");
    } finally {
      if (aliveRef.current) setTyping(false);
    }
  }, [draft, typing, messages, pollUntilReady]);

  const clear = useCallback(() => {
    setMessages([]);
    setNotice(null);
    setPhase("idle");
  }, []);

  const minutes = Math.floor(wakeSeconds / 60);
  const seconds = String(wakeSeconds % 60).padStart(2, "0");

  return (
    <div className="ui ui-panel ui-chat morris-chat" data-phase={phase}>
      <div className="ui-head">
        <img
          className="morris-chat__avatar"
          src="/assets/img/photo-f130244616-500.webp"
          alt=""
          width="24"
          height="24"
          data-source="stand-in"
          data-subject="William Morris"
        />
        <div className="ui-title">William Morris</div>
        <span className="ui-chat__online">
          {phase === "ready" ? "At his desk" : phase === "waking" ? "Waking" : "Asleep"}
        </span>
        <span className="ui-spacer"></span>
        {messages.length > 0 && (
          <button className="morris-chat__clear" type="button" onClick={clear}>
            Clear the conversation
          </button>
        )}
      </div>

      <div className="ui-chat__body morris-chat__body" ref={bodyRef} role="log" aria-live="polite">
        {phase === "idle" && (
          <div className="morris-chat__start">
            <p className="morris-chat__start-line">
              He sleeps between visitors, so the machine costs nothing while nobody talks. Waking
              him takes a minute or two.
            </p>
            <button className="btn btn--primary" type="button" onClick={begin}>
              Begin the conversation
              <svg className="m-arrow" viewBox="0 0 12 12" aria-hidden="true">
                <path className="m-arrow__stem" d="M1 6h8" />
                <path className="m-arrow__head" d="M4.5 2.5 8 6l-3.5 3.5" />
              </svg>
            </button>
          </div>
        )}
        {phase === "waking" && (
          <div className="morris-chat__start" role="status">
            <span className="morris-chat__dots" aria-hidden="true">
              <i></i>
              <i></i>
              <i></i>
            </span>
            <p className="morris-chat__start-line">
              Waking him. The first visit takes a minute or two; after that he answers in seconds.
            </p>
            <p className="morris-chat__elapsed">
              {minutes}:{seconds} waited
            </p>
          </div>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div className="ui-msg ui-msg--me" key={i}>
              <span className="ui-msg__text">{m.content}</span>
            </div>
          ) : (
            <div className="ui-msg ui-msg--bot" key={i}>
              <img
                className="morris-chat__avatar ui-msg__av"
                src="/assets/img/photo-f130244616-500.webp"
                alt=""
                width="22"
                height="22"
              />
              <span className="ui-msg__col">
                <span className="ui-chat__answer">{m.content}</span>
              </span>
            </div>
          ),
        )}
        {typing && (
          <div className="ui-msg ui-msg--bot" aria-hidden="true">
            <img
              className="morris-chat__avatar ui-msg__av"
              src="/assets/img/photo-f130244616-500.webp"
              alt=""
              width="22"
              height="22"
            />
            <span className="ui-msg__col">
              <span className="morris-chat__dots">
                <i></i>
                <i></i>
                <i></i>
              </span>
            </span>
          </div>
        )}
        {notice && (
          <p className="morris-chat__notice" role="status">
            {notice}
          </p>
        )}
      </div>

      <form
        className="morris-chat__inputrow"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <label className="visually-hidden" htmlFor="morris-draft">
          Your message to William Morris
        </label>
        <textarea
          id="morris-draft"
          ref={inputRef}
          className="morris-chat__input"
          rows={1}
          placeholder="Write to him"
          value={draft}
          disabled={phase !== "ready"}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button
          className="morris-chat__send"
          type="submit"
          disabled={phase !== "ready" || !draft.trim() || typing}
          aria-label="Send the message"
        >
          <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
            <path
              d="M8 13V3.5M3.8 7.5 8 3.3l4.2 4.2"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </form>
    </div>
  );
}
