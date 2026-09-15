import React from "react";
import {render,screen} from "@testing-library/react";
import {afterEach,describe,it,expect,vi} from "vitest";
vi.mock("@/components/auth-context",()=>({useAuth:()=>({session:null,me:null})}));
import Pricing from "./page";

const plan = {price_usd:1,interval:"month",credits_per_cycle:5,test_mode:true,configured:true,billing_enabled:true,guest_quick_limit:1,user_quick_daily_limit:10,free_lifetime_credits:2};

describe("pricing",()=>{
  afterEach(()=>vi.unstubAllGlobals());
  it("keeps purchase available when billing is enabled",async()=>{
    vi.stubGlobal("fetch",vi.fn().mockResolvedValue({ok:true,json:async()=>plan}));
    render(<Pricing/>);
    expect(await screen.findByText(/TEST \/ DEMO PRICING/)).toBeInTheDocument();
    expect(screen.getByRole("link",{name:"Sign in for test checkout"})).toHaveAttribute("href","/sign-in");
  });
  it("clearly disables paid subscriptions while preserving free plan details",async()=>{
    vi.stubGlobal("fetch",vi.fn().mockResolvedValue({ok:true,json:async()=>({...plan,configured:false,billing_enabled:false})}));
    render(<Pricing/>);
    expect(await screen.findByText(/PAID SUBSCRIPTIONS ARE TEMPORARILY UNAVAILABLE/)).toBeInTheDocument();
    expect(screen.getByRole("button",{name:"Subscriptions unavailable"})).toBeDisabled();
    expect(screen.getByText(/Free accounts and existing credits remain available/)).toBeInTheDocument();
    expect(screen.getAllByText("Verified free").length).toBeGreaterThan(0);
    expect(screen.getByText(/Failed analyses return reserved credits/)).toBeInTheDocument();
    expect(screen.getByText(/Credits are granted once and do not reset/)).toBeInTheDocument();
  });
});
