"use client";
import {useState, type FormEvent} from "react";
import Link from "next/link";
import {useRouter} from "next/navigation";
import {supabase} from "@/lib/supabase";

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
    if (result.error) setError(result.error.message);
    else if (mode === "sign-up") setMessage("Check your email to verify the account. Your two lifetime credits are granted after verification.");
    else router.push("/analyze");
  }
  return <form className="auth-form" onSubmit={submit}><label>Email<input name="email" type="email" autoComplete="email" required /></label>
    <label>Password<input name="password" type="password" autoComplete={mode === "sign-up" ? "new-password" : "current-password"} minLength={8} required /></label>
    {error && <p className="alert error" role="alert">{error}</p>}{message && <p className="alert success" role="status">{message}</p>}
    <button className="button primary" disabled={busy}>{busy ? "Please wait…" : mode === "sign-up" ? "Create account" : "Sign in"}</button>
    <p>{mode === "sign-up" ? "Already registered?" : "New here?"} <Link href={mode === "sign-up" ? "/sign-in" : "/sign-up"}>{mode === "sign-up" ? "Sign in" : "Create an account"}</Link></p>
  </form>;
}
