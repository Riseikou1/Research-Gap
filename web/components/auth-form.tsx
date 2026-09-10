"use client";
import {useState, type FormEvent} from "react";
import Link from "next/link";
import {useRouter} from "next/navigation";
import {supabase} from "@/lib/supabase";
import {PasswordField} from "./password-field";

export function friendlyAuthError(message: string) {
  const value = message.toLowerCase();
  if (value.includes("invalid login") || value.includes("invalid credentials")) return "The email or password is incorrect.";
  if (value.includes("email not confirmed")) return "Verify your email before signing in.";
  if (value.includes("already registered") || value.includes("already exists")) return "An account already exists for this email. Try signing in.";
  if (value.includes("password") && (value.includes("weak") || value.includes("characters"))) return "Choose a stronger password with at least 8 characters.";
  if (value.includes("rate") || value.includes("too many")) return "Too many attempts. Please wait a little and try again.";
  return "Authentication is temporarily unavailable. Please try again.";
}

export function AuthForm({mode}: {mode: "sign-in" | "sign-up"}) {
  const [error, setError] = useState(""); const [message, setMessage] = useState(""); const [busy, setBusy] = useState(false);
  const router = useRouter();
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError(""); setBusy(true);
    const client = supabase();
    if (!client) { setError("Authentication is not configured. Follow the owner setup guide."); setBusy(false); return; }
    const data = new FormData(event.currentTarget); const email = String(data.get("email")); const password = String(data.get("password"));
    const result = mode === "sign-up" ? await client.auth.signUp({email, password, options: {emailRedirectTo: `${location.origin}/profile`}}) : await client.auth.signInWithPassword({email, password});
    setBusy(false);
    if (result.error) setError(friendlyAuthError(result.error.message));
    else if (mode === "sign-up") setMessage("Check your email to verify the account. Your two lifetime credits are granted after verification.");
    else router.push("/analyze");
  }
  return <form className="auth-form panel" onSubmit={submit} aria-busy={busy}><label>Email<input name="email" type="email" autoComplete="email" required disabled={busy}/></label>
    <PasswordField name="password" label="Password" autoComplete={mode === "sign-up" ? "new-password" : "current-password"} disabled={busy}/>
    {error && <p className="alert error" role="alert">{error}</p>}{message && <p className="alert success" role="status">{message}</p>}
    <button className="button primary" disabled={busy}>{busy ? "Please wait…" : mode === "sign-up" ? "Create account" : "Sign in"}</button>
    {mode === "sign-up" && <p className="credit-note"><strong>Two lifetime Full Gap Analysis credits</strong><span>Verified free accounts receive this allowance once. Credits do not renew or reset.</span></p>}
    <aside className="auth-switch" aria-label="Switch authentication page"><span>{mode === "sign-up" ? "Already registered?" : "New to Research GAP?"}</span> <Link href={mode === "sign-up" ? "/sign-in" : "/sign-up"}>{mode === "sign-up" ? "Sign in" : "Create your free account"}</Link></aside>
  </form>;
}
