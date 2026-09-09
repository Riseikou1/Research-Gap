"use client";
import Link from "next/link";
import {useEffect, useState} from "react";
import {API_URL, post} from "@/lib/api";
import {useAuth} from "@/components/auth-context";

type Plan = {price_usd: number; interval: string; credits_per_cycle: number; test_mode: boolean};
export default function Pricing() {
  const {session, me} = useAuth();
  const [plan, setPlan] = useState<Plan | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {fetch(`${API_URL}/billing/plan`).then(response => response.json()).then(setPlan).catch(() => setError("Plan details are unavailable."));}, []);
  async function open(path: string) {
    setError("");
    try {
      const value = await post(path, session?.access_token) as {url?: string};
      if (value.url) location.assign(value.url);
    } catch (caught) {setError(caught instanceof Error ? caught.message : "Payment action unavailable");}
  }
  return <div className="page"><div className="page-head"><div><p className="eyebrow">Simple test allowance</p><h1>Usage & pricing</h1><p>One analysis credit means one submitted Full Gap Analysis—not each internal search or model call.</p></div>{me?.signed_in && <div><strong>{me.credits}</strong><br/><span className="hint">credits remaining</span></div>}</div>
    <article className="panel pricing-card"><p className="test-banner">TEST / DEMO PRICING — NOT APPROVED FOR LIVE SALES</p><h2>Researcher placeholder</h2><p className="price">{plan ? `$${plan.price_usd}` : "…"} <small>/ month</small></p><p>{plan?.credits_per_cycle ?? "…"} Full Gap Analysis credits after each successfully paid billing cycle. Credits are granted by a verified Stripe webhook, never by the browser return page.</p><ul><li>{plan?.credits_per_cycle ?? "…"} full-analysis credits per paid cycle</li><li>Optional full text still uses one credit</li><li>Manage cancellation in Stripe’s customer portal</li></ul>{error && <p className="alert error" role="alert">{error}</p>}{me?.signed_in ? <div className="actions"><button className="button primary" onClick={() => open("/billing/checkout")}>Open test Checkout</button><button className="button secondary" onClick={() => open("/billing/portal")}>Manage subscription</button></div> : <Link className="button primary" href="/sign-up">Create a verified account</Link>}</article>
    <div className="limitation"><strong>Free accounts</strong><p>Each verified free account receives exactly two lifetime Full Gap Analysis credits. They do not reset monthly. Guests receive one Quick Search in a rolling 24-hour window.</p></div></div>;
}
