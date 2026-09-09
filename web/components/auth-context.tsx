"use client";
import {createContext, useContext, useEffect, useState, type ReactNode} from "react";
import type {Session} from "@supabase/supabase-js";
import {getMe, type Me} from "@/lib/api";
import {supabase} from "@/lib/supabase";

type AuthState = {session: Session | null; me: Me | null; loading: boolean; refresh: () => Promise<void>};
const Context = createContext<AuthState>({session: null, me: null, loading: true, refresh: async () => {}});
export function AuthProvider({children}: {children: ReactNode}) {
  const [session, setSession] = useState<Session | null>(null);
  const [me, setMe] = useState<Me | null>(null);
  const [loading, setLoading] = useState(true);
  async function refresh(current: Session | null = session) {
    try { setMe(await getMe(current?.access_token)); } catch { setMe(null); }
  }
  useEffect(() => {
    const client = supabase();
    if (!client) { getMe().then(setMe).catch(() => setMe(null)).finally(() => setLoading(false)); return; }
    client.auth.getSession().then(({data}) => {
      setSession(data.session); return getMe(data.session?.access_token);
    }).then(setMe).catch(() => setMe(null)).finally(() => setLoading(false));
    const {data} = client.auth.onAuthStateChange((_event, next) => {
      setSession(next); getMe(next?.access_token).then(setMe).catch(() => setMe(null));
    });
    return () => data.subscription.unsubscribe();
  }, []);
  return <Context.Provider value={{session, me, loading, refresh: () => refresh()}}>{children}</Context.Provider>;
}
export const useAuth = () => useContext(Context);
