import React from "react";import {render,screen}from"@testing-library/react";import{describe,it,expect,vi}from"vitest";
vi.mock("./auth-context",()=>({useAuth:()=>({me:{signed_in:true,role:"admin"},loading:false})}));
vi.mock("@/lib/supabase",()=>({supabase:()=>null}));
import {Header} from "./header";
describe("authenticated navigation",()=>{it("shows role-gated admin and account navigation",()=>{render(<Header/>);expect(screen.getByRole("link",{name:"Admin"})).toBeInTheDocument();expect(screen.getByRole("link",{name:"Profile"})).toBeInTheDocument()})});
