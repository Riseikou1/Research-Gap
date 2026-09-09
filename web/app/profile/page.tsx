"use client";
import {useState, type FormEvent} from "react";
import Link from "next/link";
import {API_URL, post} from "@/lib/api";
import {useAuth} from "@/components/auth-context";
import {supabase} from "@/lib/supabase";

export default function Profile() {
  const {session, me, refresh} = useAuth();
  const [message, setMessage] = useState(""); const [error, setError] = useState("");
  if (!me?.signed_in) return <div className="page"><h1>Profile</h1><p>Sign in to manage your profile and data.</p><Link className="button primary" href="/sign-in">Sign in</Link></div>;
  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError(""); const data = new FormData(event.currentTarget);
    try {
      const response = await fetch(`${API_URL}/profile`, {method:"PATCH", headers:{"Content-Type":"application/json", Authorization:`Bearer ${session?.access_token}`}, body:JSON.stringify({display_name:String(data.get("name"))})});
      if (!response.ok) throw new Error(); await refresh(); setMessage("Profile updated.");
    } catch {setError("Profile could not be updated.");}
  }
  async function avatar(event: FormEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0]; if (!file) return;
    const response = await fetch(`${API_URL}/profile/avatar`, {method:"POST", headers:{Authorization:`Bearer ${session?.access_token}`, "Content-Type":file.type}, body:file});
    if (!response.ok) setError((await response.json()).detail ?? "Avatar upload failed");
    else {await refresh(); setMessage("Avatar updated.");}
  }
  async function claim() {
    const value = await post("/account/claim-guest", session?.access_token) as {claimed:number};
    setMessage(`${value.claimed} guest analyses claimed.`);
  }
  async function removeAccount() {
    if (!window.confirm("Delete your research analyses, profile, and avatar? This cannot be undone.")) return;
    const response = await fetch(`${API_URL}/account`, {method:"DELETE", headers:{Authorization:`Bearer ${session?.access_token}`}});
    if (!response.ok) {setError((await response.json()).detail ?? "Account deletion failed."); return;}
    await supabase()?.auth.signOut(); location.assign("/");
  }
  return <div className="page"><div className="page-head"><div><p className="eyebrow">Account</p><h1>Your profile</h1><p>{me.verified ? "Email verified" : "Email verification required for full analyses"} · {me.credits} credits</p></div>{me.profile?.avatar_url && <img src={me.profile.avatar_url} alt="Profile avatar" width="88" height="88"/>}</div>
    <form className="panel auth-form" onSubmit={save}><label>Display name<input name="name" defaultValue={me.profile?.display_name ?? ""} minLength={1} maxLength={80} required/></label><label>Profile picture<input type="file" accept="image/png,image/jpeg,image/webp" onChange={avatar}/><small>PNG, JPEG, or WebP; maximum 2 MB. The server checks file signatures.</small></label>{message && <p className="alert success" role="status">{message}</p>}{error && <p className="alert error" role="alert">{error}</p>}<div className="actions"><button className="button primary">Save profile</button><button className="button secondary" type="button" onClick={claim}>Claim this browser’s guest history</button></div></form>
    <div className="limitation"><strong>Account deletion</strong><p>Research analyses and avatars are deleted. Minimal anonymized usage/payment audit records may be retained for reconciliation.</p><button className="button secondary" onClick={removeAccount}>Delete my account data</button></div></div>;
}
