import React from "react";
import {fireEvent, render, screen, waitFor} from "@testing-library/react";
import {beforeEach, describe, expect, it, vi} from "vitest";

const signInWithPassword=vi.fn();const updateUser=vi.fn();const refresh=vi.fn();
vi.mock("@/lib/supabase",()=>({supabase:()=>({auth:{signInWithPassword,updateUser,signOut:vi.fn()}})}));
vi.mock("@/components/auth-context",()=>({useAuth:()=>({
  session:{access_token:"token",user:{email:"researcher@example.test",email_confirmed_at:"2026--01-T00:00:00Z"}},
  me:{signed_in:true,verified:true,credits:2,role:"user",plan_label:"Free",subscription_status:"none",profile:{display_name:"Researcher",avatar_url:null,email:"researcher@example.test"}},
  refresh,sessionLoading:false,profileLoading:false,
})}));
import Profile from "./page";

describe("profile",()=>{
  beforeEach(()=>{vi.clearAllMocks();signInWithPassword.mockResolvedValue({error:null});updateUser.mockResolvedValue({error:null});});
  it("shows identity, verification, plan, and credits",()=>{render(<Profile/>);expect(screen.getByText("researcher@example.test")).toBeInTheDocument();expect(screen.getByText("Verified")).toBeInTheDocument();expect(screen.getByText(/Free · none/)).toBeInTheDocument();expect(screen.getByText("2")).toBeInTheDocument()});
  it("validates matching passwords before reauthentication",()=>{render(<Profile/>);fireEvent.change(screen.getByLabelText("Current password"),{target:{value:"old-password"}});fireEvent.change(screen.getByLabelText("New password"),{target:{value:"new-password"}});fireEvent.change(screen.getByLabelText("Confirm new password"),{target:{value:"different"}});fireEvent.click(screen.getByRole("button",{name:"Change password"}));expect(screen.getByRole("alert")).toHaveTextContent("do not match");expect(signInWithPassword).not.toHaveBeenCalled()});
  it("reauthenticates, updates, confirms, and clears password fields",async()=>{render(<Profile/>);const current=screen.getByLabelText("Current password");const next=screen.getByLabelText("New password");const confirm=screen.getByLabelText("Confirm new password");fireEvent.change(current,{target:{value:"old-password"}});fireEvent.change(next,{target:{value:"new-password"}});fireEvent.change(confirm,{target:{value:"new-password"}});fireEvent.click(screen.getByRole("button",{name:"Change password"}));await waitFor(()=>expect(screen.getByRole("status")).toHaveTextContent("Password changed successfully"));expect(signInWithPassword).toHaveBeenCalledWith({email:"researcher@example.test",password:"old-password"});expect(updateUser).toHaveBeenCalledWith({password:"new-password"});expect(screen.getByLabelText("Current password")).toHaveValue("")});
});
