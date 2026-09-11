"use client";
import {useCallback, useEffect, useState, type FormEvent} from "react";
import Link from "next/link";
import {API_URL, post} from "@/lib/api";
import {useAuth} from "@/components/auth-context";

type User = {user_id:string; email:string|null; display_name:string; role:string; status:string; subscription_status:string; balance:number};
type Summary = {user_count:number; analyses:Array<{status:string;mode:string;count:number}>; failed_payment_events:Array<Record<string,unknown>>; audit_log:Array<Record<string,unknown>>};

export default function Admin() {
  const {session, me, loading} = useAuth();
  const [users, setUsers] = useState<User[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState("");
  const load = useCallback(async () => {
    const headers = {Authorization: `Bearer ${session?.access_token}`};
    const [summaryResponse, usersResponse] = await Promise.all([
      fetch(`${API_URL}/admin/summary`, {headers}), fetch(`${API_URL}/admin/users`, {headers})
    ]);
    if (!summaryResponse.ok || !usersResponse.ok) throw new Error("Administrator data is unavailable.");
    setSummary(await summaryResponse.json()); setUsers(await usersResponse.json());
  }, [session?.access_token]);
  useEffect(() => {if (me?.role === "admin") void load().catch(caught => setError(caught.message));}, [me?.role, load]);
  async function adjust(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); const data = new FormData(event.currentTarget);
    try {
      await post("/admin/credits", session?.access_token, {user_id:String(data.get("user")), amount:Number(data.get("amount")), reason:String(data.get("reason"))});
      await load(); event.currentTarget.reset();
    } catch (caught) {setError(caught instanceof Error ? caught.message : "Adjustment failed");}
  }
  async function changeStatus(user: User) {
    const next = user.status === "suspended" ? "active" : "suspended";
    const reason = window.prompt(`Reason to mark this account ${next}:`);
    if (!reason) return;
    try {await post(`/admin/users/${encodeURIComponent(user.user_id)}/status`, session?.access_token, {status:next, reason}); await load();}
    catch (caught) {setError(caught instanceof Error ? caught.message : "Status change failed");}
  }
  if (loading) return <div className="page">Checking authorization…</div>;
  if (me?.role !== "admin") return <div className="page"><h1>Administrator only</h1><p>This route requires a server-stored administrator role.</p><Link href="/">Return home</Link></div>;
  return <div className="page"><div className="page-head"><div><p className="eyebrow">Owner operations</p><h1>Admin dashboard</h1><p>Account, usage, analysis, subscription, webhook, and audit summaries. Private research text is omitted from this broad view.</p></div><strong>{summary?.user_count ?? 0} users</strong></div>
    {error && <p className="alert error" role="alert">{error}</p>}
    <div className="result-grid">{summary?.analyses.map((item,index) => <div className="metric" key={index}><strong>{item.count}</strong><span>{item.mode} {item.status}</span></div>)}</div>
    <section className="panel"><h2>Accounts</h2><div className="data-scroll"><table className="data-table"><thead><tr><th>User</th><th>Status</th><th>Plan</th><th>Credits</th><th>Action</th></tr></thead><tbody>{users.map(user => <tr key={user.user_id}><td>{user.display_name || user.email || user.user_id}</td><td>{user.status}</td><td>{user.subscription_status}</td><td>{user.balance}</td><td><button className="button secondary" onClick={() => changeStatus(user)}>{user.status === "suspended" ? "Reactivate" : "Suspend"}</button></td></tr>)}</tbody></table></div></section>
    <section><h2>Adjust credits</h2><form className="form-row" onSubmit={adjust}><label>User<select name="user" required>{users.map(user => <option key={user.user_id} value={user.user_id}>{user.email ?? user.user_id}</option>)}</select></label><label>Amount<input name="amount" type="number" min="-1000" max="1000" required/></label><label>Audit reason<input name="reason" minLength={3} maxLength={500} required/></label><button className="button primary">Record adjustment</button></form></section>
    <section><h2>Failed webhooks</h2>{summary?.failed_payment_events.length ? <pre>{JSON.stringify(summary.failed_payment_events,null,2)}</pre> : <p>No failed webhook records. Inspect server logs and use Stripe’s test-mode resend control when retry is needed.</p>}<h2>Admin audit log</h2><pre>{JSON.stringify(summary?.audit_log ?? [],null,2)}</pre></section>
  </div>;
}
