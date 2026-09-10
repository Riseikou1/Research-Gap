"use client";
import {createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode} from "react";
import type {Session} from "@supabase/supabase-js";
import {getMe, type Me} from "@/lib/api";
import {supabase} from "@/lib/supabase";

type AuthState = {session: Session | null; me: Me | null; sessionLoading: boolean; profileLoading: boolean; loading: boolean; refresh: () => Promise<void>};
const Context = createContext<AuthState>({session: null, me: null, sessionLoading: true, profileLoading: true, loading: true, refresh: async () => {}});
export function AuthProvider({children}: {children: ReactNode}) {
  const [session, setSession] = useState<Session | null>(null);
  const [me, setMe] = useState<Me | null>(null);
  const [sessionLoading, setSessionLoading] = useState(true);
  const [profileLoading, setProfileLoading] = useState(true);
  const requestId = useRef(0);
  const inFlightUser = useRef<string | null>(null);
  const refreshFor = useCallback(async (current: Session | null, force = false) => {
    const userKey = current?.user.id ?? "guest";
    if (!force && inFlightUser.current === userKey) return;
    inFlightUser.current = userKey;
    const id = ++requestId.current; setProfileLoading(true);
    try { const next = await getMe(current?.access_token); if (id === requestId.current) setMe(next); }
    catch { if (id === requestId.current) setMe(null); }
    finally { if (inFlightUser.current === userKey) inFlightUser.current = null; if (id === requestId.current) setProfileLoading(false); }
  }, []);
  useEffect(() => {
    const client = supabase();
    if (!client) { setSessionLoading(false); void refreshFor(null); return; }
    client.auth.getSession().then(({data}) => {
      setSession(data.session); setSessionLoading(false); void refreshFor(data.session);
    }).catch(() => {setSession(null); setSessionLoading(false); setProfileLoading(false);});
    const {data} = client.auth.onAuthStateChange((event, next) => {
      setSession(next); setSessionLoading(false);
      if (event === "SIGNED_OUT") {requestId.current++; setMe(null); setProfileLoading(false); return;}
      void refreshFor(next);
    });
    return () => data.subscription.unsubscribe();
  }, [refreshFor]);
  return <Context.Provider value={{session, me, sessionLoading, profileLoading, loading: sessionLoading, refresh: () => refreshFor(session, true)}}>{children}</Context.Provider>;
}
export const useAuth = () => useContext(Context);
