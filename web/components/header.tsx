"use client";
import Link from "next/link";
import {useAuth} from "./auth-context";
import {supabase} from "@/lib/supabase";

export function Header() {
  const {me, loading} = useAuth();
  return <header className="site-header"><Link className="brand" href="/"><span>RG</span> Research GAP</Link>
    <nav aria-label="Primary navigation">
      <Link href="/analyze">New analysis</Link><Link href="/history">History</Link><Link href="/pricing">Pricing</Link>
      {me?.role === "admin" && <Link href="/admin">Admin</Link>}
      {!loading && (me?.signed_in ? <><Link href="/profile">Profile</Link><button className="link-button" onClick={() => supabase()?.auth.signOut()}>Sign out</button></> : <Link href="/sign-in">Sign in</Link>)}
    </nav>
  </header>;
}
