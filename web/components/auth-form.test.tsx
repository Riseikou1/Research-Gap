import React from "react";
import {fireEvent, render, screen, waitFor} from "@testing-library/react";
import {beforeEach, describe, expect, it, vi} from "vitest";

const push = vi.fn(); const signInWithPassword = vi.fn(); const signUp = vi.fn();
vi.mock("next/navigation", () => ({useRouter: () => ({push})}));
vi.mock("@/lib/supabase", () => ({supabase: () => ({auth:{signInWithPassword,signUp}})}));
import {AuthForm} from "./auth-form";

describe("AuthForm", () => {
  beforeEach(() => {vi.clearAllMocks(); signInWithPassword.mockResolvedValue({error:null}); signUp.mockResolvedValue({error:null});});
  it("has an independent accessible non-submit password visibility control", () => {
    render(<AuthForm mode="sign-in"/>); const input=screen.getByLabelText("Password"); const toggle=screen.getByRole("button",{name:"Show password"});
    expect(toggle).toHaveAttribute("type","button"); expect(input).toHaveAttribute("type","password"); fireEvent.click(toggle);
    expect(input).toHaveAttribute("type","text"); expect(screen.getByRole("button",{name:"Hide password"})).toBeInTheDocument();
  });
  it("makes sign-up navigation prominent", () => {render(<AuthForm mode="sign-in"/>);expect(screen.getByRole("link",{name:"Create your free account"})).toHaveAttribute("href","/sign-up");});
  it("shows loading and maps invalid credentials safely", async () => {signInWithPassword.mockImplementation(async()=>{await Promise.resolve();return{error:{message:"Invalid login credentials: provider detail"}}});render(<AuthForm mode="sign-in"/>);fireEvent.change(screen.getByLabelText("Email"),{target:{value:"a@example.test"}});fireEvent.change(screen.getByLabelText("Password"),{target:{value:"password1"}});fireEvent.click(screen.getByRole("button",{name:"Sign in"}));expect(screen.getByRole("button",{name:"Please wait…"})).toBeDisabled();await waitFor(()=>expect(screen.getByRole("alert")).toHaveTextContent("email or password is incorrect"));expect(screen.queryByText(/provider detail/)).not.toBeInTheDocument();});
  it("explains the non-recurring verified allowance on sign-up",()=>{render(<AuthForm mode="sign-up"/>);expect(screen.getByText(/Credits do not renew or reset/)).toBeInTheDocument();expect(screen.getByRole("link",{name:"Sign in"})).toHaveAttribute("href","/sign-in");});
});
