import React from "react";
import {render,screen} from "@testing-library/react";
import {describe,it,expect,vi} from "vitest";
vi.mock("@/components/auth-context",()=>({useAuth:()=>({session:null,me:null})}));
import Pricing from "./page";
describe("pricing",()=>{it("compares real features and identifies placeholder test pricing",()=>{vi.stubGlobal("fetch",vi.fn(()=>Promise.reject(new Error("offline"))));render(<Pricing/>);expect(screen.getByText(/TEST \/ DEMO PRICING/)).toBeInTheDocument();expect(screen.getAllByText("Guest").length).toBeGreaterThan(0);expect(screen.getAllByText("Verified free").length).toBeGreaterThan(0);expect(screen.getAllByText("Paid researcher").length).toBeGreaterThan(0);expect(screen.getByText(/Failed analyses return reserved credits/)).toBeInTheDocument();expect(screen.getByText(/Credits are granted once and do not reset/)).toBeInTheDocument()})});
