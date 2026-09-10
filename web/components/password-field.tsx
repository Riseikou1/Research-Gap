"use client";

import {useState} from "react";

export function PasswordField({
  name, label, autoComplete, minLength = 8, disabled = false,
}: {name: string; label: string; autoComplete: string; minLength?: number; disabled?: boolean}) {
  const [visible, setVisible] = useState(false);
  const controlLabel = `${visible ? "Hide" : "Show"} ${label.toLowerCase()}`;
  return <label>{label}<span className="password-control">
    <input name={name} type={visible ? "text" : "password"} autoComplete={autoComplete}
      minLength={minLength} required disabled={disabled}/>
    <button type="button" className="password-toggle" aria-label={controlLabel}
      title={controlLabel} aria-pressed={visible} onClick={() => setVisible(value => !value)} disabled={disabled}>
      {visible ? <EyeOffIcon/> : <EyeIcon/>}
    </button>
  </span></label>;
}

function EyeIcon() {
  return <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z"/><circle cx="12" cy="12" r="2.5"/></svg>;
}
function EyeOffIcon() {
  return <svg aria-hidden="true" viewBox="0 0 24 24"><path d="m3 3 18 18M10.6 6.1A10 10 0 0 1 12 6c6 0 9.5 6 9.5 6a16 16 0 0 1-2.1 2.8M6.2 6.2C3.8 8 2.5 12 2.5 12s3.5 6 9.5 6c1.4 0 2.7-.3 3.8-.8M9.9 9.9a3 3 0 0 0 4.2 4.2"/></svg>;
}
