"use client";
import {useState, type FormEvent} from "react";
import {useRouter} from "next/navigation";
import {createAnalysis} from "@/lib/api";
import {useAuth} from "@/components/auth-context";

const example = "How can retrieval-augmented generation reduce hallucinations in multilingual medical question answering?";
export default function AnalyzePage() {
  const {session, me, loading} = useAuth(); const router = useRouter();
  const [mode,setMode] = useState<"quick"|"full">("quick"); const [error,setError]=useState(""); const [busy,setBusy]=useState(false);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();setError(""); const data=new FormData(event.currentTarget); const research_idea=String(data.get("idea") ?? "").trim();
    if (research_idea.length < 10) {setError("Describe the research idea in at least 10 characters.");return;}
    setBusy(true);
    try {const created=await createAnalysis({research_idea,mode,paper_limit:Number(data.get("limit")),full_text:mode === "full" && data.get("full_text") === "on"},session?.access_token); router.push(`/analyses/${created.analysis_id}`);}
    catch (caught) {setError(caught instanceof Error ? caught.message : "Analysis could not start.");setBusy(false);}
  }
  const adminExempt = me?.credit_exempt === true;
  const fullAllowed = Boolean(me?.signed_in && me.verified && (adminExempt || me.credits > 0));
  return <div className="page"><div className="page-head"><div><p className="eyebrow">New investigation</p><h1>Start with the research idea.</h1><p>Be specific about the problem, method, population, data, outcome, and constraints when they matter.</p></div>{me?.signed_in && <div><strong>{adminExempt ? "Admin access" : me.credits}</strong><br/><span className="hint">{adminExempt ? "account credits are not charged" : "full-analysis credits"}</span></div>}</div>
    <form className="panel" onSubmit={submit}><label className="field">Research idea<textarea name="idea" minLength={10} maxLength={5000} required aria-describedby="idea-help" placeholder={example}/><small id="idea-help">Example: {example}</small></label>
      <fieldset><legend>Analysis depth</legend><div className="mode-grid"><label className="mode"><input type="radio" name="mode" checked={mode==="quick"} onChange={()=>setMode("quick")}/><strong>Quick Search</strong><span>Retrieval and basic deterministic ranking. One guest search per rolling 24 hours; no paid AI or gap verification.</span></label>
        <label className="mode"><input type="radio" name="mode" checked={mode==="full"} onChange={()=>setMode("full")}/><strong>Full Gap Analysis</strong><span>Evidence extraction, landscape analysis, candidate gaps, and verification. {adminExempt ? "Administrator analyses do not use account credits." : "Uses one credit."}</span></label></div></fieldset>
      {mode === "full" && adminExempt && <p className="alert" role="status">Administrator exemption applies only to internal account credits. Provider calls still have real cost, and normal safety limits remain active.</p>}
      {mode === "full" && !loading && !fullAllowed && <p className="alert error" role="alert">{!me?.signed_in ? "Sign in to use a Full Gap Analysis. Verified free accounts receive exactly two lifetime credits." : !me.verified ? "Verify your email before using a full-analysis credit." : "No full-analysis credits remain. Review the test plan on Pricing."}</p>}
      <details><summary>Advanced options</summary><div className="form-row"><label>Paper limit<select name="limit" defaultValue="20"><option value="10">10 papers</option><option value="20">20 papers</option><option value="40">40 papers</option></select></label>
        {mode === "full" && <label><span><input name="full_text" type="checkbox"/> Use available full text</span><small>{adminExempt ? "Does not use an internal account credit, but takes longer and may increase real provider cost." : "Still costs one credit, but takes longer and may increase provider cost."} Only declared open-access locations are attempted.</small></label>}</div></details>
      {error && <p className="alert error" role="alert">{error}</p>}<button className="button primary" disabled={busy || (mode === "full" && !fullAllowed)}>{busy ? "Starting…" : mode === "quick" ? "Run Quick Search" : "Run Full Gap Analysis"}</button>
    </form></div>;
}
