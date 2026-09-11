"use client";
import {useState} from "react";
import Link from "next/link";
import {useAuth} from "./auth-context";
import {supabase} from "@/lib/supabase";

export function Header() {
  const {session, me, sessionLoading} = useAuth(); const [open, setOpen] = useState(false);
  const signedIn = Boolean(session);
  return <header className="site-header"><div className="header-inner"><Link className="brand" href="/"><span>RG</span> Research GAP</Link>
    <button className="nav-toggle" type="button" aria-expanded={open} aria-controls="primary-navigation" onClick={() => setOpen(value => !value)}><span className="sr-only">Toggle navigation</span><i/><i/><i/></button>
    <nav id="primary-navigation" data-open={open} aria-label="Primary navigation">
      <Link href="/analyze">Analyze</Link>{signedIn && <Link href="/history">History</Link>}<Link href="/about">About</Link><Link href="/pricing">Pricing</Link>
      {me?.role === "admin" && <Link href="/admin">Admin</Link>}
      <span className="account-nav" aria-live="polite">{sessionLoading ? <span className="account-skeleton" aria-label="Loading account"/> : signedIn ? <><Link href="/profile">Account</Link><button className="link-button" onClick={() => supabase()?.auth.signOut()}>Sign out</button></> : <Link className="nav-sign-in" href="/sign-in">Sign in</Link>}</span>
    </nav>
  </div></header>;
}
