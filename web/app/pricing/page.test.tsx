import React from "react";import{render,screen}from"@testing-library/react";import{describe,it,expect,vi}from"vitest";
vi.mock("@/components/auth-context",()=>({useAuth:()=>({session:null,me:null})}));
import Pricing from "./page";
describe("pricing",()=>{it("always identifies placeholder test pricing",()=>{vi.stubGlobal("fetch",vi.fn(()=>Promise.reject(new Error("offline"))));render(<Pricing/>);expect(screen.getByText(/TEST \/ DEMO PRICING/)).toBeInTheDocument()})});
