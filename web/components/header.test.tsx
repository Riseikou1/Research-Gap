import React from "react";
import {render,screen} from "@testing-library/react";
import {describe,it,expect,vi} from "vitest";
const state:{session:object|null;me:{signed_in:boolean;role:string};sessionLoading:boolean}={session:{user:{id:"u"}},me:{signed_in:true,role:"admin"},sessionLoading:false};
vi.mock("./auth-context",()=>({useAuth:()=>state}));
vi.mock("@/lib/supabase",()=>({supabase:()=>null}));
import {Header} from "./header";
describe("authenticated navigation",()=>{
  it("shows role-gated navigation from the session without waiting for profile",()=>{render(<Header/>);expect(screen.getByRole("link",{name:"Admin"})).toBeInTheDocument();expect(screen.getByRole("link",{name:"Account"})).toBeInTheDocument();expect(screen.queryByRole("link",{name:"Sign in"})).not.toBeInTheDocument()});
  it("reserves account space while session resolves",()=>{state.session=null;state.sessionLoading=true;const{container}=render(<Header/>);expect(screen.getByLabelText("Loading account")).toBeInTheDocument();expect(container.querySelector(".account-nav")).toBeInTheDocument()});
});
