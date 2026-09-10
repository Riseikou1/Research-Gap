"use client";

import Image from "next/image";
import Link from "next/link";
import {useState, type FormEvent} from "react";
import {API_URL, post, safeApiMessage} from "@/lib/api";
import {useAuth} from "@/components/auth-context";
import {PasswordField} from "@/components/password-field";
import {friendlyAuthError} from "@/components/auth-form";
import {supabase} from "@/lib/supabase";

export default function Profile() {
  const {session, me, refresh, sessionLoading, profileLoading} = useAuth();
  const [message, setMessage] = useState(""); const [error, setError] = useState("");
  const [passwordMessage, setPasswordMessage] = useState(""); const [passwordError, setPasswordError] = useState("");
  const [passwordBusy, setPasswordBusy] = useState(false); const [passwordResetKey, setPasswordResetKey] = useState(0);
  const [confirmation, setConfirmation] = useState(""); const [deleting, setDeleting] = useState(false);
  const email = session?.user.email ?? me?.profile?.email ?? "Email unavailable";

  if (sessionLoading) return <div className="page narrow-page"><p role="status">Loading your account…</p></div>;
  if (!session) return <div className="page narrow-page"><h1>Your account</h1><p>Sign in to manage your profile and data.</p><Link className="button primary" href="/sign-in">Sign in</Link></div>;

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError(""); setMessage(""); const data = new FormData(event.currentTarget);
    try {
      const response = await fetch(`${API_URL}/profile`, {method:"PATCH", headers:{"Content-Type":"application/json", Authorization:`Bearer ${session?.access_token}`}, body:JSON.stringify({display_name:String(data.get("name"))})});
      if (!response.ok) throw new Error(); await refresh(); setMessage("Profile updated.");
    } catch {setError("Profile could not be updated. Please try again.");}
  }
  async function avatar(event: FormEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0]; if (!file) return; setError("");
    try {
      const response = await fetch(`${API_URL}/profile/avatar`, {method:"POST", headers:{Authorization:`Bearer ${session?.access_token}`, "Content-Type":file.type}, body:file});
      if (!response.ok) throw new Error(); await refresh(); setMessage("Avatar updated.");
    } catch { setError("The avatar could not be uploaded. Use a PNG, JPEG, or WebP under 2 MB."); }
  }
  async function claim() {
    setError("");
    try { const value = await post("/account/claim-guest", session?.access_token) as {claimed:number}; setMessage(`${value.claimed} guest analyses claimed.`); }
    catch { setError("Guest history could not be claimed right now."); }
  }
  async function changePassword(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setPasswordError(""); setPasswordMessage("");
    const data = new FormData(event.currentTarget); const current = String(data.get("current_password") ?? "");
    const next = String(data.get("new_password") ?? ""); const confirm = String(data.get("confirm_password") ?? "");
    if (next.length < 8) {setPasswordError("Your new password must contain at least 8 characters."); return;}
    if (next !== confirm) {setPasswordError("The new passwords do not match."); return;}
    if (current === next) {setPasswordError("Choose a new password that differs from your current password."); return;}
    const client = supabase(); if (!client || !session?.user.email) {setPasswordError("Password changes are temporarily unavailable."); return;}
    setPasswordBusy(true);
    const reauthenticated = await client.auth.signInWithPassword({email: session.user.email, password: current});
    if (reauthenticated.error) {setPasswordBusy(false); setPasswordError(friendlyAuthError(reauthenticated.error.message)); return;}
    const updated = await client.auth.updateUser({password: next}); setPasswordBusy(false);
    if (updated.error) {setPasswordError(friendlyAuthError(updated.error.message)); return;}
    setPasswordResetKey(value => value + 1); setPasswordMessage("Password changed successfully.");
  }
  async function manageSubscription() {
    setError(""); try { const value = await post("/billing/portal", session?.access_token) as {url?: string}; if (value.url) location.assign(value.url); }
    catch { setError("The subscription portal is unavailable right now."); }
  }
  async function removeAccount() {
    if (confirmation !== "DELETE") return; setDeleting(true); setError("");
    try {
      const response = await fetch(`${API_URL}/account`, {method:"DELETE", headers:{Authorization:`Bearer ${session?.access_token}`}});
      if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(safeApiMessage(response.status, typeof body.detail === "string" ? body.detail : "")); }
      await supabase()?.auth.signOut(); location.assign("/");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Account deletion is temporarily unavailable."); setDeleting(false); }
  }

  const activeSubscription = ["active", "trialing", "past_due", "unpaid", "incomplete"].includes(me?.subscription_status ?? "none");
  return <div className="page account-page"><div className="page-head"><div><p className="eyebrow">Account</p><h1>Your research profile</h1><p>Identity, allowance, subscription, and privacy controls in one place.</p></div>{me?.profile?.avatar_url ? <Image className="avatar" src={me.profile.avatar_url} alt="Profile avatar" width={96} height={96} unoptimized/> : <div className="avatar avatar-placeholder" aria-label="No profile avatar">{email.slice(0,1).toUpperCase()}</div>}</div>
    <section className="account-summary" aria-label="Account summary">
      <div><span>Email</span><strong>{email}</strong></div><div><span>Verification</span><strong>{session.user.email_confirmed_at || me?.verified ? "Verified" : "Not verified"}</strong></div>
      <div><span>Display name</span><strong>{me?.profile?.display_name || "Not set"}</strong></div><div><span>Role</span><strong>{me?.role ?? "User"}</strong></div>
      <div><span>Credits</span><strong>{profileLoading ? "…" : me?.credits ?? 0}</strong></div><div><span>Plan</span><strong>{me?.plan_label ?? "Free"} · {me?.subscription_status ?? "No subscription"}</strong></div>
    </section>

    <section className="settings-section"><div className="section-heading"><h2>Profile details</h2><p>Choose the name and image shown in your account.</p></div><form className="panel auth-form" onSubmit={save}><label>Display name<input name="name" defaultValue={me?.profile?.display_name ?? ""} minLength={1} maxLength={80} required/></label><label>Profile picture<input type="file" accept="image/png,image/jpeg,image/webp" onChange={avatar}/><small>PNG, JPEG, or WebP; maximum 2 MB.</small></label>{message && <p className="alert success" role="status">{message}</p>}<div className="actions"><button className="button primary">Save profile</button><button className="button secondary" type="button" onClick={claim}>Claim guest history</button></div></form></section>

    <section className="settings-section"><div className="section-heading"><h2>Change password</h2><p>For your security, we verify your current password first. Passwords go only to Supabase Authentication.</p></div><form key={passwordResetKey} className="panel auth-form" onSubmit={changePassword} aria-busy={passwordBusy}><PasswordField name="current_password" label="Current password" autoComplete="current-password" disabled={passwordBusy}/><PasswordField name="new_password" label="New password" autoComplete="new-password" disabled={passwordBusy}/><PasswordField name="confirm_password" label="Confirm new password" autoComplete="new-password" disabled={passwordBusy}/>{passwordError && <p className="alert error" role="alert">{passwordError}</p>}{passwordMessage && <p className="alert success" role="status">{passwordMessage}</p>}<button className="button primary" disabled={passwordBusy}>{passwordBusy ? "Changing password…" : "Change password"}</button></form></section>

    <section className="danger-zone"><p className="eyebrow">Danger zone</p><h2>Delete account</h2><p>This permanently removes your Supabase sign-in identity, profile, avatar, and saved analyses. Minimal pseudonymous anti-abuse and required payment/audit records may remain.</p>{activeSubscription && <div className="alert error"><strong>Subscription action required</strong><p>Manage or cancel your paid subscription before deletion so billing cannot continue.</p><button className="button secondary" type="button" onClick={manageSubscription}>Manage subscription</button></div>}<label>Type <strong>DELETE</strong> to confirm<input value={confirmation} onChange={event => setConfirmation(event.target.value)} autoComplete="off"/></label>{error && <p className="alert error" role="alert">{error}</p>}<button className="button danger" type="button" disabled={confirmation !== "DELETE" || deleting || activeSubscription} onClick={removeAccount}>{deleting ? "Deleting account…" : "Permanently delete account"}</button></section>
  </div>;
}
