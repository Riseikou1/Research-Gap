"use client";

import Link from "next/link";
import {useCallback, useEffect, useRef, useState} from "react";
import {getHistory, type Analysis} from "@/lib/api";
import {useAuth} from "@/components/auth-context";
import {StatusPill} from "@/components/status-pill";
import {HISTORY_INVALIDATED_EVENT} from "@/lib/history-events";

type HistoryState = "session" | "signed_out" | "loading" | "loaded" | "error";

export default function HistoryPage() {
  const {session, sessionLoading} = useAuth();
  const [items, setItems] = useState<Analysis[]>([]);
  const [state, setState] = useState<HistoryState>("session");
  const [error, setError] = useState("");
  const requestId = useRef(0);
  const controller = useRef<AbortController | null>(null);
  const userId = session?.user.id ?? null;
  const token = session?.access_token ?? null;

  const load = useCallback(() => {
    if (!userId || !token) return;
    controller.current?.abort();
    const currentController = new AbortController();
    controller.current = currentController;
    const id = ++requestId.current;
    setState("loading");
    setError("");
    void getHistory(token, currentController.signal).then(value => {
      if (id !== requestId.current || currentController.signal.aborted) return;
      setItems(value);
      setState("loaded");
    }).catch(caught => {
      if (id !== requestId.current || currentController.signal.aborted) return;
      setError(caught instanceof Error ? caught.message : "History is temporarily unavailable.");
      setState("error");
    });
  }, [token, userId]);

  useEffect(() => {
    requestId.current += 1;
    controller.current?.abort();
    if (sessionLoading) {
      setState("session");
      return;
    }
    if (!userId || !token) {
      setItems([]);
      setState("signed_out");
      return;
    }
    load();
    return () => controller.current?.abort();
  }, [load, sessionLoading, token, userId]);

  useEffect(() => {
    const refresh = () => {
      if (!sessionLoading && userId && token) load();
    };
    window.addEventListener(HISTORY_INVALIDATED_EVENT, refresh);
    return () => window.removeEventListener(HISTORY_INVALIDATED_EVENT, refresh);
  }, [load, sessionLoading, token, userId]);

  return <div className="page"><div className="page-head"><div><p className="eyebrow">Private workspace</p><h1>Analysis history</h1><p>Only analyses owned by this account appear here.</p></div><Link className="button primary" href="/analyze">New analysis</Link></div>
    {(state === "session" || state === "loading") && <HistoryLoading session={state === "session"}/>}
    {state === "signed_out" && <div className="panel history-state"><h2>Sign in to view history</h2><p className="hint">Guest Quick Search results remain available from their result link, but private history belongs to signed-in accounts.</p><Link className="button secondary" href="/sign-in">Sign in</Link></div>}
    {state === "error" && <div className="panel history-state" role="alert"><h2>History could not be loaded</h2><p>{error}</p><button className="button secondary" type="button" onClick={load}>Try again</button></div>}
    {state === "loaded" && items.length === 0 && <div className="panel history-state"><h2>No analyses yet</h2><p className="hint">Run a Quick Search or Full Gap Analysis to begin.</p></div>}
    {state === "loaded" && items.length > 0 && <div className="history-list">{items.map(item=><Link className="history-item" href={`/analyses/${item.analysis_id}`} key={item.analysis_id}><div><h3>{item.research_idea}</h3><p>{item.mode === "quick" ? "Quick Search" : "Full Gap Analysis"} · {new Date(item.created_at).toLocaleString()}</p></div><StatusPill status={item.status}/></Link>)}</div>}
  </div>;
}

function HistoryLoading({session}: {session: boolean}) {
  return <div className="panel history-state history-loading" aria-busy="true" role="status"><h2>{session ? "Restoring your session…" : "Loading your analyses…"}</h2><span/><span/><span/></div>;
}
