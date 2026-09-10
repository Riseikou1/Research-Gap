"use client";
import Link from "next/link";
import {useEffect, useState} from "react";
import {API_URL, post} from "@/lib/api";
import {useAuth} from "@/components/auth-context";

type Plan = {price_usd: number; interval: string; credits_per_cycle: number; test_mode: boolean; configured: boolean; guest_quick_limit: number; user_quick_daily_limit: number; free_lifetime_credits: number};
const rows = [
  ["Quick Search", "Included", "Included", "Included"],
  ["Rolling Quick Search limit", "{guest} / 24h", "{user} / 24h", "{user} / 24h"],
  ["Paper retrieval and basic ranking", "Yes", "Yes", "Yes"],
  ["Saved analysis history", "Browser session", "Yes", "Yes"],
  ["Structured evidence extraction", "—", "With credit", "With credit"],
  ["Landscape and gap generation", "—", "With credit", "With credit"],
  ["Targeted verification", "—", "With credit", "With credit"],
  ["Open full-text mode", "—", "When available", "When available"],
];
export default function Pricing() {
  const {session, me} = useAuth(); const [plan, setPlan] = useState<Plan | null>(null); const [error, setError] = useState(""); const [busy, setBusy] = useState("");
  useEffect(() => {const controller = new AbortController(); fetch(`${API_URL}/billing/plan`, {signal:controller.signal}).then(response => {if(!response.ok) throw new Error(); return response.json();}).then(setPlan).catch(caught => {if(caught instanceof Error && caught.name !== "AbortError") setError("Current plan details are temporarily unavailable.");}); return () => controller.abort();}, []);
  async function open(path: string) {setError("");setBusy(path);try {const value=await post(path,session?.access_token) as {url?:string};if(value.url) location.assign(value.url);}catch {setError("Billing is temporarily unavailable. Please try again.");setBusy("");}}
  const replaceLimits=(value:string)=>value.replace("{guest}",String(plan?.guest_quick_limit??1)).replace("{user}",String(plan?.user_quick_daily_limit??10));
  return <div className="page pricing-page"><div className="page-head"><div><p className="eyebrow">Plans and allowances</p><h1>Choose the depth you need.</h1><p>One credit means one submitted Full Gap Analysis, including its bounded search, evidence extraction, landscape, gap candidates, and verification. Failed analyses return reserved credits.</p></div>{me?.signed_in&&<div className="credit-balance"><strong>{me.credits}</strong><span>credits remaining</span></div>}</div>
    <p className="test-banner" role="note">TEST / DEMO PRICING — NOT APPROVED FOR LIVE SALES</p>
    <div className="plan-cards">
      <article><p className="eyebrow">Explore</p><h2>Guest</h2><p className="plan-price">Free</p><p>Quick Search for paper retrieval and basic deterministic ranking. No evidence extraction, landscape, gap generation, verification, or full text.</p><strong>{plan?.guest_quick_limit??1} Quick Search / rolling 24 hours</strong></article>
      <article><p className="eyebrow">Build a history</p><h2>Verified free</h2><p className="plan-price">Free</p><p>Everything in Guest, saved history, and <strong>{plan?.free_lifetime_credits??2} lifetime</strong> Full Gap Analysis credits after email verification.</p><p>Credits are granted once and do not reset.</p><Link className="button secondary" href="/sign-up">Create free account</Link></article>
      <article className="featured-plan"><p className="eyebrow">Placeholder plan</p><h2>Paid researcher</h2><p className="plan-price">{plan?`$${plan.price_usd}`:"…"}<small> / {plan?.interval??"month"}</small></p><p><strong>{plan?.credits_per_cycle??"…"} credits</strong> after each successfully paid test billing cycle, plus saved history and full-analysis features.</p><p>Cancel or manage the test subscription through Stripe’s customer portal. Final price and credit amount will be chosen before live billing.</p>{session?<div className="actions"><button className="button primary" disabled={Boolean(busy)} onClick={()=>open("/billing/checkout")}>{busy==="/billing/checkout"?"Opening…":"Open test checkout"}</button><button className="button secondary" disabled={Boolean(busy)} onClick={()=>open("/billing/portal")}>Manage subscription</button></div>:<Link className="button primary" href="/sign-in">Sign in for test checkout</Link>}</article>
    </div>
    {error&&<p className="alert error" role="alert">{error}</p>}
    <div className="comparison-wrap"><table className="comparison-table"><caption>Feature comparison</caption><thead><tr><th scope="col">Capability</th><th scope="col">Guest</th><th scope="col">Verified free</th><th scope="col">Paid researcher</th></tr></thead><tbody>{rows.map(row=><tr key={row[0]}>{row.map((cell,index)=><td key={`${index}-${cell}`} data-label={index?["","Guest","Verified free","Paid researcher"][index]:undefined}>{replaceLimits(cell)}</td>)}</tr>)}</tbody></table></div>
    <div className="limitation"><strong>Full text depends on access</strong><p>When open full text is unavailable, analysis falls back to abstracts or metadata and reports that coverage honestly. It does not claim every paper was fully read.</p></div>
  </div>;
}
